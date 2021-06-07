import skimage.io as io
import os
import skimage.transform as trans
import numpy as np

def predictGenerator(path,target_size = (256,256),as_gray = False):
    img = io.imread(path,as_gray=as_gray)
    img = img / 255
    img = trans.resize(img, target_size)
    img = np.reshape(img, (1,)+img.shape)
    yield img