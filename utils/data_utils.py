from io_utils import imgread
from img_utils import imgresize
from hash_utils import dhash, phash
import os
import numpy as np

from progressbar import ProgressBar
from skimage import io
import shutil
import Image
Image.LOAD_TRUNCATED_IMAGES = True
import hashlib
import binascii

def change_filepaths(file_paths, old, new):
  """
    Given a list of image paths, it replaces the substring 'old' with
    'new'.

    file_paths  : list of image paths
    old         : substring to be replaced
    new         : substring to be appended
  """
  new_file_paths = []
  for count,file_path in enumerate(file_paths):
      new_file_paths.append(file_path.replace(old,new))
  return new_file_paths

def get_data_paths(root_path, ext = '*.jpg'):
    """
      Given the root path, it finds all items with the given extension recursively.
      For correct outputs, each class of images should be in a separate folder in the
      given root_path. Each unique folder names defines the label of its images.

      root_path : root_path to start searching
      ext       : wildcard to define desired files. exp. *.jpg is all jpg files

      OUTPUTS---

      matches   : image paths of found items
      classes   : numeric class labels for the images.
      class_names   : class names as the including folder name
    """

    import os
    import fnmatch
    matches = []
    classes = []
    class_names = []
    for root, dirnames, filenames in os.walk(root_path):
      for filename in fnmatch.filter(filenames, ext):
          matches.append(os.path.join(root, filename))
          class_name =  os.path.basename(os.path.dirname(os.path.join(root, filename)))
          if class_name not in class_names:
               class_names.append(class_name)
          classes.append(class_names.index(class_name))

    print "There are ",len(matches), " files're found!!"
    return matches, classes, class_names

def load_data(root_path, ext):
    """
        Loads all numpy readable files on the given root path
        and ending with the given extesnion into a data matrix.
        It returns
            X : data matrix
            classes : labels
            class_names : string values of labels as folder name
            file_paths : returns full file paths for expecting sanity check
    """

    file_paths, classes, class_names = get_data_paths(root_path, ext)
    print file_paths[0]
    if 'npy' in ext:
      feat_vec = np.load(file_paths[0])
    else:
      feat_vec = np.loadtxt(file_paths[0], delimiter='\n')

    print feat_vec.shape
    X = np.zeros([len(file_paths), feat_vec.shape[0]])
    print X.shape
    #counter  = 0;
    # pb = ProgressBar(maxval= len(file_paths))
    for count, file_path in enumerate(file_paths):
        if 'npy' in ext:
          feat_vec = np.load(file_path)
        else:
          feat_vec = np.loadtxt(file_path, delimiter='\n')
        # print file_path
        X[count,:] = feat_vec
        # pb.update(count)
    # pb.finish()
    return X, np.array(classes)[:, None], class_names, file_paths

def load_imgs(img_paths, size=None):

  if size is not None:
    if None in size:
      print "Resizing only supports squared sized. (height == width)"
      return

  img_list = []
  pb = ProgressBar(maxval = len(img_paths))
  c = 0
  for img_path in img_paths:
    img = imgread(img_path)
    if size is not None:
      try:
        img = imgresize(img, size)
      except:
        print img_path
        shutil.move(img_path, '/media/retina18/SAMSUNG/PinterestImgs/RottenImages')
      try:
        assert img.shape[0] == size[0]
        assert img.shape[1] == size[1]
      except:
        print img_path
        print img.shape
        return
    img_list.append(img)
    pb.update(c)
    c +=1
  pb.finish()
  return img_list


def remove_duplicate_images(paths, ext = 'jpg'):
  '''
  Given a list of path it discovers duplicate images recursively
  and removes
  '''
  from collections import Counter
  img_dict = Counter()
  img_path_dict = {}
  duplicate_count = 0
  for path in paths:
        try:
          img = imgread(path)
        except IOError:
          print path
          os.remove(path)
          continue

        if img is None:
          print path
          os.remove(path)
          continue

        hash_val = str(list(dhash(img)))
        if img_dict[hash_val] == 0:
            img_dict[hash_val] += 1
            img_path_dict[hash_val] = path
        else:
            os.remove(path)
            print 'DUPLICATE WITH !! -- ', img_path_dict[hash_val], ' -- ', path
            print 'REMOVED!! -- ', path
            duplicate_count += 1
  print duplicate_count, ' images are REMOVED!!'

def convert_img_to_jpg(img_path):
  # if str it is img path
  try:
    img = Image.open(img_path)
    file_path, file_ext = os.path.splitext(img_path)
  except IOError:
    print 'Cannot read !! ',img_path
    return

  try:
    bg = Image.new("RGB", img.size, (255,255,255))
    bg.paste(img,img)
    bg.save(img_path.replace(file_ext,'.jpg'))
  except ValueError: # if no transparency layer
    img.convert('RGB').save(img_path.replace(file_ext,'.jpg'))
    print "Cannot convert img to jpg !!"
    return
  os.remove(img_path)

def convert_imgs_to_jpg(paths):
  """
    Given the path of images, it converts all to jpg files
  """
  for img_path in paths:
    file_path, file_ext = os.path.splitext(img_path)
    if file_ext.lower() not in [".jpg"]:
      try:
        img = Image.open(img_path)
      except IOError:
        os.remove(img_path)
        print 'DELETED !! ',img_path
        continue

      try:
        bg = Image.new("RGB", img.size, (255,255,255))
        bg.paste(img,img)
        bg.save(img_path.replace(file_ext,'.jpg'))
      except ValueError: # if no transparency layer
        img.convert('RGB').save(img_path.replace(file_ext,'.jpg'))
      os.remove(img_path)

def remove_rotten_imgs(img_paths, verbose=False):
  """
    Removes images cannot be read
  """
  remove_count = 0
  for img_path in img_paths:
      try:
          img = io.imread(img_path)
      except(IOError):
          if verbose:
            print img_path, ' DELETED !!!!'
          os.remove(img_path)
          remove_count += 1
  print "In total ", remove_count, " images deleted !!!"

def name_img(img):
  """
    Create a unique name for the given image as its hexadecimal
    match of the dhash value
  """
  str_list = ''.join(str(e) for e in list(dhash(img)[0]))
  return str(hex(int(str_list,2)))

def rename_imgs(img_paths, hash_func = 'dhash'):
  for img_path in img_paths:
    file_name = os.path.basename(img_path)
    file_name, file_ext = file_name.split('.')
    img = imgread(img_path)
    new_file_name = str(phash(img)[0])
    new_path = img_path.replace(file_name,new_file_name)
    # print new_path
    try:
      os.rename(img_path, new_path)
    except:
      print img_path
