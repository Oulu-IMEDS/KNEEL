# -*- coding: utf-8 -*-

import os, math, re
import numpy as np
import scipy
import scipy.misc as misc
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sklearn.neighbors import NearestNeighbors

from sklearn.utils import shuffle
from skimage import color, measure, morphology
from numba import jit, autojit
import random, shutil

import cv2

from .generic_utils import get_from_module


def get(identifier):
    return get_from_module(identifier, globals(), 'local_utils')


def mkdirs(folders, erase=False):
    if type(folders) is not list:
        folders = [folders]
    for fold in folders:
        if not os.path.exists(fold):
            os.makedirs(fold)
        else:
            if erase:
                shutil.rmtree(fold)
                os.makedirs(fold)

# Replace pixel values with specific color
def change_val(img,val, len, x_min, y_min, x_max, y_max):
    left_len  = (len-1)//2
    right_len = (len-1) - left_len
    row_size, col_size = img.shape[0:2]
    for le in range(-left_len, right_len + 1):
        y_min_ = max(0, y_min + le )
        x_min_ = max(0, x_min + le )

        y_max_ = min(row_size, y_max - le )
        x_max_ = min(col_size, x_max - le )

        img[y_min_:y_max_, x_min_:x_min_+1] = val
        img[y_min_:y_min_+1, x_min_:x_max_] = val
        img[y_min_:y_max_, x_max_:x_max_+1] = val
        img[y_max_:y_max_+1, x_min_:x_max_] = val
    return img


# Overlay bbox
def overlay_bbox(img, bbox, len=1, rgb=(255, 0, 0)):
    for bb in bbox:
        x_min_, y_min_, x_max_, y_max_ = bb
        x_min_, y_min_, x_max_, y_max_ = int(x_min_),int( y_min_), int(x_max_), int(y_max_)
        img[:,:,0] = change_val(img[:,:,0], rgb[0], len, x_min_, y_min_, x_max_, y_max_)
        img[:,:,1] = change_val(img[:,:,1], rgb[1], len,  x_min_, y_min_, x_max_, y_max_)
        img[:,:,2] = change_val(img[:,:,2], rgb[2], len,  x_min_, y_min_, x_max_, y_max_)
    return img


def writeImg(array, savepath):
    scipy.misc.imsave(savepath, array)


def imread(imgfile):
    assert os.path.exists(imgfile), '{} does not exist!'.format(imgfile)
    srcBGR = cv2.imread(imgfile)
    destRGB = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
    return destRGB


def imresize(img, resizeratio=1):
    '''Take care of cv2 reshape squeeze behevaior'''
    if resizeratio == 1:
        return img
    outshape = ( int(img.shape[1] * resizeratio) , int(img.shape[0] * resizeratio))
    temp = cv2.resize(img, outshape).astype(float)
    #temp = misc.imresize(img, size=outshape).astype(float)
    if len(img.shape) == 3 and img.shape[2] == 1:
        temp = np.reshape(temp, temp.shape + (1,))
    return temp


def imshow(img, size=None):
    if size is not None:
        plt.figure(figsize = size)
    else:
        plt.figure()
    plt.imshow(img)
    plt.show()


def normalize_img(X):
    min_, max_ = np.min(X), np.max(X)
    X = (X - min_)/ (max_ - min_ + 1e-9)
    X = X*255
    return X.astype(np.uint8)


def Indexflow(Totalnum, batch_size, random=True):
    numberofchunk = int(Totalnum + batch_size - 1)// int(batch_size)   # the floor
    #Chunkfile = np.zeros((batch_size, row*col*channel))
    totalIndx = np.arange(Totalnum).astype(np.int)
    if random is True:
        totalIndx = np.random.permutation(totalIndx)

    chunkstart = 0
    for chunkidx in range(int(numberofchunk)):
        thisnum = min(batch_size, Totalnum - chunkidx*batch_size)
        thisInd = totalIndx[chunkstart: chunkstart + thisnum]
        chunkstart += thisnum
        yield thisInd


def getfileinfo(imgdir, contourextList, ImgExtList, LabelExt, test_mode = False):
    '''return a list of dictionary {'thisfile':os.path.join(imgdir,f), 'thismatfile':thismatfile}
    '''
    alllist  = [f for f in os.listdir(imgdir)]
    alllist = sorted(alllist)

    returnList = []
    for f in alllist:
        if os.path.isfile(os.path.join(imgdir,f)) and \
                   os.path.splitext(f)[1] in ImgExtList:
            if test_mode is False:
                flag = 0
                for contourext in contourextList:
                    thismatfile  = os.path.join(imgdir,os.path.splitext(f)[0] + contourext + LabelExt)
                    if os.path.isfile(thismatfile):
                        returnList.append({'thisfile':os.path.join(imgdir,f), 'thismatfile':thismatfile})
                        flag = 1
                        break
                if flag == 0:
                    print(("Image: {s} does not have matfile".format(s = os.path.splitext(f)[0] )))
            else:
                returnList.append({'thisfile':os.path.join(imgdir,f), 'thismatfile': None})
    return returnList


def split_img(img, windowsize=1000, board = 0, fixed_window = False, step_size = None, tuple_slice = False):
    '''
    img dimension: channel, row, col
    output:
        (IndexDict, PackList)
        IndexDict is a dictionry, the key is the actual patch size, the values is the list of identifier,
        PackList: list of (thisPatch,org_slice ,extract_slice, thisSize,identifier), the index of Packlist
        corresponds to the identifier.
        org_slice: slice coordinated at the orignal image.
        extract_slice: slice coordinate at the extract thisPatch,
        the length of org_slice should be equal to extract_slice.

        fixed_window: if true, it forces the extracted patches has to be of size window_size.
                      we don't pad the original image to make mod(imgsize, windowsize)==0, instead, if the remaining is small,
                      we expand the left board to lefter to compensate the smaller reminging patches.

                      The default behavior is False: get all window_size patches, and collect the remining patches as it is.

        step_size: if step_size is smaller than (windowsize-2*board), we extract the patches with overlapping.
                which means the org_slice is overlapping.

    eg:
    lenght = 17
    img = np.arange(2*lenght*lenght).reshape(2,lenght,lenght)

    nm = np.zeros(img.shape).astype(np.int)

    AllDict, PackList =  split_img(img, windowsize=7, board = 0, step_size= 2,fixed_window = True)

    print img

    print '---------------------------------------'

    print AllDict.keys()

    for key in AllDict.keys():
        iden_list = AllDict[key]
        for iden in iden_list:
            thispatch = PackList[iden][0]
            org_slice = PackList[iden][1]
            extract_slice = PackList[iden][2]

            nm[:,org_slice[0],org_slice[1]] = thispatch[:,extract_slice[0],extract_slice[1]]
            print thispatch[:,extract_slice[0],extract_slice[1]]
    print nm
    print sum(nm-img)
    '''
    IndexDict = {}
    identifier = -1
    PackList = []
    row_size, col_size = img.shape[1], img.shape[2]
    if windowsize is not None and  type(windowsize) is int:
        windowsize = (windowsize, windowsize)

    if windowsize is None or (row_size <= windowsize[0] and col_size<=windowsize[1] and (not fixed_window)):
        pad_img = img
        rowsize, colsize = pad_img.shape[1:]

        org_slice = (slice(0, rowsize), slice(0, colsize))
        extract_slice = org_slice
        crop_patch_slice = (slice(0, rowsize), slice(0, colsize))
        thisSize =  (rowsize, colsize )
        identifier = identifier + 1

        org_slice_tuple = (0, 0)
        if thisSize in IndexDict:
           IndexDict[thisSize].append(identifier)
        else:
           IndexDict[thisSize] = []
           IndexDict[thisSize].append(identifier)
        PackList.append((crop_patch_slice, org_slice ,extract_slice, thisSize,identifier, org_slice_tuple))

    else:

        hidden_windowsize = (windowsize[0]-2*board, windowsize[1]-2*board)
        for each_size in hidden_windowsize:
            if each_size <= 0:
                raise RuntimeError('windowsize can not be smaller than board*2.')

        if type(step_size) is int:
            step_size = (step_size, step_size)
        if step_size is None:
            step_size = hidden_windowsize

        numRowblocks = int(math.ceil(float(row_size)/step_size[0]))
        numColblocks = int(math.ceil(float(col_size)/step_size[1]))

        # sanity check, make sure the image is at least of size window_size to the left-hand side if fixed_windows is true
        # which means,    -----*******|-----, left to the vertical board of original image is at least window_size.
        row_addition_board, col_addition_board = 0, 0
        addition_board = 0
        if fixed_window:
            if row_size + 2 * board < windowsize[0]: # means we need to add more on board.
                row_addition_board = windowsize[0] - (row_size + 2 * board )
            if col_size + 2 * board < windowsize[1]: # means we need to add more on board.
                col_addition_board = windowsize[1] - (col_size + 2 * board)
            addition_board = row_addition_board if row_addition_board > col_addition_board else col_addition_board

        left_pad = addition_board + board
        pad4d = ((0,0),( left_pad , board), ( left_pad , board ))
        pad_img = np.pad(img, pad4d, 'symmetric').astype(img.dtype)

        thisrowstart, thiscolstart =0, 0
        thisrowend, thiscolend = 0,0
        for row_idx in range(numRowblocks):
            thisrowlen = min(hidden_windowsize[0], row_size - row_idx * step_size[0])
            row_step_len = min(step_size[0], row_size - row_idx * step_size[0])

            thisrowstart = 0 if row_idx == 0 else thisrowstart + step_size[0]

            thisrowend = thisrowstart + thisrowlen

            row_shift = 0
            if fixed_window:
                if thisrowlen < hidden_windowsize[0]:
                    row_shift = hidden_windowsize[0] - thisrowlen

            for col_idx in range(numColblocks):
                thiscollen = min(hidden_windowsize[1], col_size -  col_idx * step_size[1])
                col_step_len = min(step_size[1], col_size - col_idx * step_size[1])

                thiscolstart = 0 if col_idx == 0 else thiscolstart + step_size[1]

                thiscolend = thiscolstart + thiscollen

                col_shift = 0
                if fixed_window:
                    # we need to shift the patch to left to make it at least windowsize.
                    if thiscollen < hidden_windowsize[1]:
                        col_shift = hidden_windowsize[1] - thiscollen

                #
                #----board----******************----board----
                #
                crop_r_start = thisrowstart - board - row_shift + left_pad
                crop_c_start = thiscolstart - board - col_shift + left_pad
                crop_r_end  =  thisrowend + board + left_pad
                crop_c_end  =  thiscolend + board + left_pad

                #we need to handle the tricky board condition
                # thispatch will be of size (:,:, windowsize+ 2*board)
                #thisPatch =  pad_img[:,crop_r_start:crop_r_end, crop_c_start:crop_c_end].copy()
                crop_patch_slice = (slice(crop_r_start, crop_r_end), slice(crop_c_start, crop_c_end))
                org_slice_tuple  = (crop_r_start-left_pad,  crop_c_start -left_pad )

                thisSize = (thisrowlen + 2*board + row_shift, thiscollen + 2*board + col_shift)


                org_slice = (slice(thisrowstart, thisrowend), slice(thiscolstart, thiscolend))
                # slice on a cooridinate of the original image
                extract_slice = (slice(board + row_shift, board + thisrowlen + row_shift),
                                slice(board + col_shift, board + col_shift + thiscollen))
                # extract on local coordinate of a patch

                identifier =  identifier +1
                PackList.append((crop_patch_slice,org_slice ,extract_slice, thisSize,identifier, org_slice_tuple))

                if thisSize in IndexDict:
                   IndexDict[thisSize].append(identifier)
                else:
                   IndexDict[thisSize] = []
                   IndexDict[thisSize].append(identifier)

    PackDict = {}
    for this_size in list(IndexDict.keys()):
        iden_list = IndexDict[this_size]
        this_len = len(iden_list)
        org_slice_list = []
        extract_slice_list = []
        slice_tuple_list  = []
        BatchData = np.zeros( (this_len, img.shape[0]) + tuple(this_size) )
        for idx, iden in enumerate(iden_list):
            crop_patch_slice = PackList[iden][0]
            BatchData[idx,...] = pad_img[:,crop_patch_slice[0],crop_patch_slice[1]].copy()
            org_slice_list.append(PackList[iden][1])
            extract_slice_list.append(PackList[iden][2])
            slice_tuple_list.append(PackList[iden][-1])

        PackDict[this_size]= (BatchData,org_slice_list,extract_slice_list, slice_tuple_list)

    return PackDict
