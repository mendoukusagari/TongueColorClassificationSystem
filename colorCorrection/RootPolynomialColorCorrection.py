import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import csv
from colorCorrection.RootPolynomial import RootPolynomial
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class RootPolynomialColorCorrection():
    def __init__(self):
        pass
    def sRGB2XYZ(self,img):
        rgb2xyz = (
            0.412391, 0.357584, 0.180481, 0,
            0.212639, 0.715169, 0.072192, 0,
            0.019331, 0.119195, 0.950532, 0
        )
        return img.convert("RGB", rgb2xyz)

    def XYZ2sRGB(self,img):
        xyz2rgb = (3.240970, -1.537383, -0.498611, 0,
                   -0.969244, 1.875968, 0.041555, 0,
                   0.055630, -0.203977, 1.056972, 0)
        return img.convert("RGB", xyz2rgb)

    def conv_sRGB2XYZ(self,rgb):
        M = np.array([[0.412391, 0.357584, 0.180481],
                      [0.212639, 0.715169, 0.072192],
                      [0.019331, 0.119195, 0.950532]])
        return np.dot(M, rgb.T).T
    def train(self,ref,source,degree=1):
        reference_raw = np.loadtxt(ref, delimiter=",", skiprows=1, usecols=(1, 2, 3))
        source_raw = np.loadtxt(source, delimiter=",", skiprows=1, usecols=(1, 2, 3))

        reference_xyz = self.conv_sRGB2XYZ(reference_raw)
        source_xyz = self.conv_sRGB2XYZ(source_raw)

        source_xyz = RootPolynomial(degree).fit(source_xyz)
        source_xyz = source_xyz[:, 1:]

        source_xyz_hm = np.append(source_xyz, np.ones((24, 1)), axis=1)

        ccm = np.linalg.pinv(source_xyz_hm).dot(reference_xyz)
        self.ccm = ccm
    def predict(self,input):
        input_img = Image.open(input, 'r').convert("RGB")
        input_img = self.sRGB2XYZ(input_img)
        input_img = np.asarray(input_img)
        width, height, dim = input_img.shape
        input_img = input_img.reshape(width * height, dim)
        input_img = RootPolynomial(3).fit(input_img)
        input_img = input_img[:, 1:]
        input_img = np.append(input_img, np.ones((width * height, 1)), axis=1)

        input_img = np.matmul(input_img, self.ccm)
        input_img[input_img > 255] = 255
        input_img[input_img < 0] = 0
        input_img = input_img.reshape(width, height, 3)

        input_img = input_img.astype(np.uint8)
        input_img = Image.fromarray(input_img, 'RGB')
        input_img = self.XYZ2sRGB(input_img)
        return input_img