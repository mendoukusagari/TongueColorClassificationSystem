import cv2
import numpy as np
from PIL import Image

def mask(image,path_mask,out):
    width,height = image.size
    imgMask = Image.open(path_mask, 'r').convert("LA")
    imgMask = imgMask.resize((int(width),int(height)), Image.ANTIALIAS)
    imgMask.save("./segmentation/tmp/msk.png")
    imgMask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)
    image = np.asarray(image)
    image = image[:,:,::-1]
    fin = cv2.bitwise_and(image, image, mask=imgMask)
    cv2.imwrite(out, fin, [cv2.IMWRITE_PNG_COMPRESSION, 0])

