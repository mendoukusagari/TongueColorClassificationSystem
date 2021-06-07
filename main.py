from utils import resize
from segmentation import segmentation
from colorCorrection import colorCorrection
from prediction import Predict
import matplotlib.pyplot as plt
from matplotlib.image import imread
import tensorflow as tf
import cv2

print(tf.version.VERSION)


def __main__():
    input_path = "./image/a2.jpg"
    corrected = colorCorrection.ColorCorrection.colorCorrectKPLSR(input_path,source="./utils/ref-note-8.csv",ref="./utils/new-ref4.csv",kernel="sigmoid",z=-4)
    resize.resize(corrected, "./tmp/img/tmp.jpg", "jpeg", (256, 256))
    segmentation.segmentation("./tmp/img/tmp.jpg", "./tmp/img/tmp.png")
    res = Predict.Predict.predict("./tmp/img/tmp.png")
    print(res)
__main__()