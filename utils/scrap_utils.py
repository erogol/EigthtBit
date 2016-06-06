'''
    This code includes some functions that helps scraping and crawling.
    It expects Selenium as the backend
'''

from data_utils import name_img_md5
from skimage import io

def write_file(url_list, file_name):
    '''
    Writes utl list to file per line. It is useful for post image donwloading
    '''
    f = open(file_name,'w')
    for url in url_list:
        if type(url) == list:
            for ur in url:
                if type(ur) == int:
                    f.write((str(ur)+' ').encode('utf8'))
                else:
                    f.write((ur+' ').encode('utf8'))
            f.write('\n')
        else:
            f.write((url+'\n').encode('utf8'))

    f.close()

def scrollDown(browser, numberOfScrollDowns):
    '''
    Performs scroll as you press down stroke of keyboard.
    '''
    body = browser.find_element_by_tag_name("body")
    while numberOfScrollDowns >=0:
        body.send_keys(Keys.PAGE_DOWN)
        numberOfScrollDowns -= 1
    return browser

def scrollToEnd(driver):
    '''
    Scrolls to the end of the page by using javascript
    '''
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")


def down_img(img_url, out_path):
    '''
    Downloads image by naming it MD5 hash
    '''
    img = io.imread(img_url)
    img_name = name_img_md5(img)
    img_name =  img_name+".jpg"
    io.imsave(out_path+img_name,img)

def down_gif(gif_url, out_path):
    '''
    Downloads gif. It also extracts the mid frame as the resulting image instead
    of whole gif file.
    '''
    gif = io.imread(gif_url)
    idx = gif.shape[0]/2
    img = gif[idx]
    img_name = name_img_md5(img)
    img_name =  img_name+".jpg"
    io.imsave(out_path+img_name,img)


set
