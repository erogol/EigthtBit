# Unicode processing functions
import re, urlparse
import errno
import os
import signal
import time
import string
import random
import urllib
from functools import wraps
from pyvirtualdisplay import Display
from selenium import webdriver
from skimage import io, transform
from utils.data_utils import name_img_md5

def urlEncodeNonAscii(b):
    return re.sub('[\x80-\xFF]', lambda c: '%%%02x' % ord(c.group(0)), b)

def iriToUri(iri):
    parts= urlparse.urlparse(iri)
    return urlparse.urlunparse(
        part.encode('idna') if parti==1 else urlEncodeNonAscii(part.encode('utf-8'))
        for parti, part in enumerate(parts)
    )

def thumbnail(img, size=150):

    from math import floor
    width, height = img.shape[1], img.shape[0]

    if width == height:
        img = transform.resize(img, [size, size])

    elif height > width:
        ratio = float(width) / float(height)
        newwidth = ratio * size
        img = transform.resize(img, [size, int(floor(newwidth))])

    elif width > height:
        ratio = float(height) / float(width)
        newheight = ratio * size
        img = transform.resize(img, [int(floor(newheight)), size])

    img = (img*255).astype('uint8')
    return img

class TimeoutError(Exception):
    pass

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

class GoogleScrapper(object):
    def __init__(self, query, num_imgs, img_out_path, html_out_path):
        self.num_imgs = num_imgs
        self.query = query.replace(" ", "+')
        self.img_out_path = img_out_path
        self.html_out_path = html_out_path

    def set_query(self,query):
        self.query = query

    def collect_urls(self):
        # Hide broswer window and proceed sliently
        display = Display(visible=0, size=(800, 600))
        display.start()

        # Create output folder if not exists
        folder_path = os.path.join(self.html_out_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Firefox
        try:
            driver = webdriver.Firefox()
        except:
            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument('--no-sandbox')
            driver = webdriver.Chrome('/usr/local/bin/chromedriver', chrome_options=chrome_options)

        # ------------------------------
        # The actual test scenario: Test the codepad.org code execution service.

        # Go to Google Image Search
        driver.get('https://www.google.com.tr/search?q='+self.query.replace(' ','%20')+'&safe=off&client=ubuntu&hs=Qb5&channel=fs&source=lnms&tbm=isch&sa=X&ei=u_blU83ZO8H5yQP8n4G4AQ&ved=0CAgQ_AUoAQ&biw=1871&bih=985#facrc=_&imgdii=_&imgrc=MlXYWUwoUzf0EM%253A%3BPUxnkCQK1e-qOM%3Bhttp%253A%252F%252Fwww.cs.bilkent.edu.tr%252F~canf%252FCS533%252FhwSpring13%252Fprojects%252FprojectPresentations%252FIMG_3151.JPG%3Bhttp%253A%252F%252Fwww.cs.bilkent.edu.tr%252F~canf%252FCS533%252FhwSpring13%252Fprojects%252FCS533projectPresentations.htm%3B2592%3B1936')

        img_count = 0;
        img_count_old = 0;
        pageSource = '';
        while img_count < self.num_imgs:
            img_count_old = img_count
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            pageSource = driver.page_source
            matches = re.findall(r'imgurl=(.*?(?:&|\.(?:jpg|gif|png|jpeg)))', pageSource, re.I)
            img_count = len(matches)
            print img_count
            time.sleep(5)
            if img_count == img_count_old:
                print 'Google returns no more images!!!'
                break;


        # Close the browser!
        driver.quit()

        # Save HTML
        f = open(os.path.join(folder_path,self.query.replace(' ','_')+'.html'),'w')
        f.write(iriToUri(pageSource))
        f.close()

    def download_imgs(self):
        '''
            Read Goolge Image Search Result Page html file and get the file urls
            We assume that we have the html file with the same name as query
        '''

        # Create Output Folders if not exist
        img_folder_path = os.path.join(self.img_out_path, self.query.replace(' ','_'))
        if not os.path.exists(img_folder_path):
            os.makedirs(img_folder_path)
        if not os.path.exists(self.img_out_path):
            os.makedirs(self.img_out_path)

        # URL check variables
        filename_length = 20
        filename_charset = string.ascii_letters + string.digits

        # Read the html file and start to process
        linestring = open(self.html_out_path+'/'+self.query+'.html').read()
        count = 1
        img_urls = []
        for match in re.findall(r'imgurl=(.*?(?:&|\.(?:jpg|gif|png|jpeg)))', linestring, re.I):
            # remove specials chars
            match = urllib.unquote(match)

            # If URL is too long ignore it.
            if len(match) <  5:
                continue;

            # Download ufolder_pathrls with wget
            try:
                filename = ''.join(random.choice(filename_charset)
                             for s in range(filename_length))
                file_ext = match.split('.')[-1]
                if file_ext.lower() not in ['svg', 'bmp']:
                    with timeout(seconds=10):
                        img = io.imread(match)
                    img_resized = thumbnail(img, size=500)
                    img_name = name_img_md5(img_resized)
                    img_out_path = os.path.join(img_folder_path,img_name +'.'+file_ext.lower())
                    io.imsave(img_out_path, img_resized)
                    count += 1
                    img_urls.append(match);
                else:
                    print 'File format is not jpeg : ', filename
            except TimeoutError:
                print "Timeout !! : ", match
            except:
                print "unable to open url " + match

        # Save img urls
        f = open(self.img_out_path+self.query+'.txt','w') # open a file to write urls with the query name
        for item in img_urls:
            f.write("%s\n" % item)
        f.close()
