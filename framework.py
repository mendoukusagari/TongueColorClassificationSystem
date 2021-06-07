from utils import resize
from segmentation import segmentation
from colorCorrection import colorCorrection
from prediction import Predict2
import matplotlib.pyplot as plt
from matplotlib.image import imread
import tensorflow as tf
import cv2
from PIL import Image
import pandas as pd
from IPython import display


def predict(path,src,ref):
    input_path = path
    fig=plt.figure()
    img = Image.open(input_path)
    fig.add_subplot(1,3,1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Before")
    
    corrected2,corrected1 = colorCorrection.ColorCorrection().colorCorrectKPLSRO(input=input_path,source=src,ref=ref)
    fig.add_subplot(1,3,2)
    plt.imshow(corrected2)
    plt.axis('off')
    plt.title("Color correction")
    
    #corrected1.save("test.jpg")
    resized = resize.resize(corrected2, (256, 256))
    segmentation.Segmentation().segment_image(corrected1, resized, "./tmp/img/tmp.png")
    out = Image.open("./tmp/img/tmp.png")
    fig.add_subplot(1, 3, 3)
    plt.imshow(out)
    plt.axis('off')
    plt.title("Image segmentation")
    
    res,res2,res3 = Predict2.Predict().predict("./tmp/img/tmp.png")
    print("===================================")
    print("RESULT")
    print("===================================")
    print("Median color LAB is "+str(res2))
    print("Dominant color is "+res3)
    
    plt.show()
    
    data = [res.values()]
    df = pd.DataFrame(data, columns=res.keys())
    print("Table percentage:")
    display.display(df)
    
    
    
    
    fig2 = plt.figure()
    plt.axis('off')
    idx=1
    color_name={"cyan":"Cyan", "red":"Red", "lp":"Light Purple", "purple":"Purple","dr":"Deep Red","lr":"Light Red","bk":"Black","gy":"Gray","w":"White","ye":"Yellow"}
    for x in ['red', 'dr', 'lr','bk', 'w', 'ye', 'purple', 'cyan', 'lp', 'gy']:
        fig2.add_subplot(2,5,idx)
        plt.imshow(Image.open("./tmp/ext/"+x+".png"))
        plt.axis('off')
        plt.title(color_name[x])
        idx+=1
   
    fig3 = plt.figure(figsize = (3.2, 3))
    ax = plt.axes(projection ="3d")
    plt.xlim(xmin=0,xmax=100)
    plt.ylim(ymin=0,ymax=70)
    ax.set_zlim(-50,50)
    ax.scatter3D(res2[0],res2[1],res2[2], color = "green",label="lab",  s=60)
    ax.legend()
    plt.title("Median point")
    plt.show()
    