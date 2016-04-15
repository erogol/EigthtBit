import numpy as np
import urllib
import dill
import cStringIO
from PIL import Image as PILImage
from multiprocessing import Pool
from skimage.io import imread, imsave

def imgread(url):
    """
    Read an image given web url or file path
    inputs:
        url - img file path or web url
    """
    img = imread(url)
    return img

def imgsave(img, img_path):
    """
    Write an image to given file path
    inputs:
        img - numpy img array
        img_path - path to write
    """
    imsave(img_path, img)

def base642img(img_base64):
    """
    Convert base64 to numpy img array
    inputs:
        img_base64 - base64 decoded string

    outputs:
        img - numpy image array
    """
    strIO = cStringIO(img_base64)
    img = io.imread(strIO)
    return img

def img2base64(img):
    """
    Creates an image embedded in HTML base64 format.
    inputs:
        img - converts numpy img array to base64 string ready to send

    outputs:
        base64_str - base64 string ready to be sent to server with image
            prefix 'data:image/jpeg;base64'
    """
    img_pil = PILImage.fromarray(img.astype('uint8'))

    # baseheight = 480
    # hpercent = (baseheight / float(image_pil.size[1]))
    # wsize = int((float(image_pil.size[0]) * float(hpercent)))
    # image_pil = image_pil.resize((wsize, baseheight), PILImage.ANTIALIAS)
    #img.save('resized_image.jpg')
    #image_pil = image_pil.resize((256, 256))
    string_buf = cStringIO.StringIO()
    img_pil.save(string_buf, format='jpeg')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    base64_str = 'data:image/jpeg;base64,' + data
    return base64_str


def save_obj(obj, file_path):
    dill.dump(obj, open(file_path,'wb'))

def load_obj(file_path):
    return dill.load(open(file_path, 'rb'))
