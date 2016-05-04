import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.color import rgb2gray
# from pathos.multiprocessing import ProcessingPool, ThreadingPool
from io_utils import imgread
from progressbar import ProgressBar
from ..cpp.wrapper import im2col
import sys

def remove_img_border(img):
    """
        Removes black image borders
    """
    non_zero_indices = np.nonzero(img)

    top_row = non_zero_indices[0].min()
    bot_row = non_zero_indices[0].max()

    top_col = non_zero_indices[1].min()
    bot_col = non_zero_indices[1].max()

    croped_img = img[top_row:bot_row, top_col:bot_col]
    return croped_img, [top_row, bot_row, top_col, bot_col]

def imgresize(img, size = [500, None]):
    """
        Img is resized by a certain height keeping aspect ratio
    """
    if None in size:
        indx = size.index(None)
        if indx == 0:
            w_size = size[1]
            r = float(w_size) / img.shape[1]
            dim = (int(img.shape[0] * r), w_size)
            img = resize(img, dim)
        elif indx == 1:
            h_size = size[0]
            r = float(h_size) / img.shape[0]
            dim = (h_size, int(img.shape[1] * r))
            img = resize(img, dim)
    else:
        img = resize(img, size)
    img = img*255.0
    assert img.max() <= 255
    return img

def imggray(img):
    return np.round(rgb2gray(img)*255.0).astype('uint8')


def create_img_grids(img, num_grids, center=False):
    """
        Creates img patch grids

        img       : 2D or 3D array of an image
        num_grids : A tuple of number of horizontal and vertical grids
        center    : Take a center grid
    """

    ver_num = num_grids[0]
    hor_num = num_grids[1]

    img_size = img.shape[0]
    img_width = img.shape[1]

    patches = []

    if num_grids != [0,0]:
        hor_patch_size = np.ceil(img_width / hor_num)
        ver_patch_size = np.floor(img_size / ver_num)

        for j in range(ver_num):
            for i in range(hor_num):
                hor_start_indx = i*hor_patch_size
                ver_start_indx = j*ver_patch_size
                patch = img[ver_start_indx:ver_start_indx+ver_patch_size, hor_start_indx : hor_start_indx + hor_patch_size]
                patches.append(patch)
    else:
        hor_patch_size = np.ceil(img_width / 2)
        ver_patch_size = np.floor(img_size / 2)

    if center == True:
        patch = img[img_size/2 - ver_patch_size/2 : img_size/2 + ver_patch_size/2, img_width/2 - hor_patch_size/2 : img_width/2 + hor_patch_size/2]
        patches.append(patch)
    return patches


def _extract_random_patch(img_path, num_patches, patch_size, img_size, gray,  count, verbose = False):
    cont_flag = True # cont until find textures patch
    if verbose:
        print 'Random Patch Extraction -- ', count, ' / ',num_patches, '\n'
        sys.stdout.flush()
    img = imgread(img_path)
    if gray:
        if verbose:
            print img.shape
        img = rgb2gray(img)
    if img_size != None:
        img  = imgresize(img, img_size)
    while cont_flag:
        row = np.random.randint(0, img.shape[0] - patch_size)
        column = np.random.randint(0, img.shape[1] - patch_size)
        if gray:
            patch = img[row:row+patch_size, column:column+patch_size]
            assert patch.ndim == 2
        else:
            patch = img[row:row+patch_size, column:column+patch_size, :]
            assert patch.shape[2] == 3
        if np.unique(patch).shape[0] > 1:
            cont_flag = False
    return patch

def extract_random_patches(image_paths, num_patches, patch_size, img_size = None, gray = True,num_proc=5):
    """
        Given the list of img paths, it generated desired number of
        random img patches. Final patches has dimension of
        num_patches , patch_size , patch_size , <channels> if gray == false
    """
    if gray:
        patches = np.ones([num_patches, patch_size, patch_size])
    else:
        patches = np.ones([num_patches, patch_size, patch_size, 3])
    num_images = len(image_paths)
    np.random.shuffle(image_paths)
    image_paths = image_paths * (num_patches / len(image_paths))
    image_paths += image_paths[0:(num_patches%len(image_paths))]

    assert len(image_paths) == num_patches

    if num_proc == 1:
        pb = ProgressBar(maxval=num_patches)
        for patch_count in range(len(image_paths)):
            img_path = image_paths[patch_count]
            patch = _extract_random_patch(img_path, num_patches, patch_size, img_size, gray, patch_count)
            patches[patch_count,:] = patch
            pb.update(patch_count)
        pb.finish()

    else:
        p = ProcessingPool(num_proc)
        patches = p.map(_extract_random_patch,
                    image_paths,
                    [num_patches]*num_patches,
                    [patch_size]*num_patches,
                    [img_size]*num_patches,
                    [gray]*num_patches,
                    range(num_patches),
                    [False]*num_patches)

    return np.array(patches)

def extract_random_patches2(image_paths, num_patches, patch_size, img_size = None, gray = True):
    num_patches_per_img = num_patches / len(image_paths)
    rem_patches =  num_patches % len(image_paths)

    if gray:
        patches = np.empty([num_patches,patch_size**2])
    else:
        patches = np.empty([num_patches,3*(patch_size**2)])
    counter = 0
    pb = ProgressBar(maxval=len(image_paths))
    for c, image_path in enumerate(image_paths):
        image = imgread(image_path)
        image = imgresize(image,img_size)
        n_channels = 3
        if gray:
            image = imggray(image)
            n_channels = 1
            assert image.max() >= 1
        else:
            assert image.max() > 1

        col_size_h = (image.shape[0]-patch_size)/1+1
        assert col_size_h != 0
        col_size_w = (image.shape[1]-patch_size)/1+1
        assert col_size_w != 0

        patches_tmp = np.empty([col_size_h,col_size_w,(patch_size**2)*n_channels], np.float64)

        if gray:
            im2col(image.astype(np.float64)[None,:,:,None], patches_tmp, patch_size, 1)
        else:
            im2col(image.astype(np.float64)[None,:], patches_tmp, patch_size, 1)

        patches_tmp = patches_tmp.reshape([col_size_h*col_size_w,(patch_size**2)*n_channels])

        if c+1 == len(image_paths):
            indices = np.random.randint(0,patches_tmp.shape[0],num_patches_per_img+rem_patches)
        else:
            indices = np.random.randint(0,patches_tmp.shape[0],num_patches_per_img)

        sel_patches = patches_tmp[indices,:]
        assert sel_patches.shape[0] > 0
        assert sel_patches.shape[1] > 0

        if c+1 == len(image_paths):
            patches[counter:counter+num_patches_per_img+rem_patches,:] = sel_patches
        else:
            patches[counter:(counter+num_patches_per_img),:] = sel_patches
            counter += num_patches_per_img
        pb.update(c)
    pb.finish()
    return patches
