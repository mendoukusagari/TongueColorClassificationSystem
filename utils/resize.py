from PIL import Image
import os, sys

def resize(im,size):
    imResize = im.resize((size[0],size[1]), Image.ANTIALIAS)
    imResize.save("./tmp/img/tmp.jpg")
    return "./tmp/img/tmp.jpg"
