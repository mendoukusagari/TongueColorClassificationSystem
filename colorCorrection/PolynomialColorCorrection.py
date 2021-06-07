import argparse
import numpy as np
import csv
from PIL import Image, ImageDraw, ImageFont
from sklearn.preprocessing import PolynomialFeatures

class PolynomialColorCorrection():
    def __init__(self):
        pass
    def gamma_table(self,gamma_r, gamma_g, gamma_b, gain_r=1.0, gain_g=1.0, gain_b=1.0):
        r_tbl = [min(255, int((x / 255.) ** (gamma_r) * gain_r * 255.)) for x in range(256)]
        g_tbl = [min(255, int((x / 255.) ** (gamma_g) * gain_g * 255.)) for x in range(256)]
        b_tbl = [min(255, int((x / 255.) ** (gamma_b) * gain_b * 255.)) for x in range(256)]
        return r_tbl + g_tbl + b_tbl

    def applyGamma(self,img, gamma=1.0):
        inv_gamma = 1. / gamma
        return img.point(self.gamma_table(inv_gamma, inv_gamma, inv_gamma))

    def deGamma(self,img, gamma=1.0):
        return img.point(self.gamma_table(gamma, gamma, gamma))
    def conv_sRGB2XYZ(self,rgb):
        M = np.array([[0.412391, 0.357584, 0.180481],
                      [0.212639, 0.715169, 0.072192],
                      [0.019331, 0.119195, 0.950532]])
        return np.dot(M, rgb.T).T

    def sRGB2XYZ(self,img):
        rgb2xyz = (
            0.412391, 0.357584, 0.180481, 0,
            0.212639, 0.715169, 0.072192, 0,
            0.019331, 0.119195, 0.950532, 0
        )
        return img.convert("RGB", rgb2xyz)
    def conv_sXYZ2RGB(self,rgb):
        M = np.array([[3.240970, -1.537383, -0.498611],
                      [-0.969244, 1.875968, 0.041555],
                      [0.055630, -0.203977, 1.056972]])
        return np.matmul(rgb, M.T)

    def XYZ2sRGB(self,img):
        xyz2rgb = (3.240970, -1.537383, -0.498611, 0,
                   -0.969244, 1.875968, 0.041555, 0,
                   0.055630, -0.203977, 1.056972, 0)
        return img.convert("RGB", xyz2rgb)
    def correctColor(self, img):
        return img.convert("RGB", tuple(self.ccm.transpose().flatten()))
    def train(self,source,ref):
        reference_raw = np.loadtxt(ref, delimiter=",", skiprows=1, usecols=(1, 2, 3))
        source_raw = np.loadtxt(source, delimiter=",", skiprows=1, usecols=(1, 2, 3))

        reference_linear = np.power(reference_raw, 1.0)
        source_linear = np.power(source_raw, 1.0)

        reference_xyz = self.conv_sRGB2XYZ(reference_linear)
        source_xyz = self.conv_sRGB2XYZ(source_linear)

        source_xyz_hm = np.append(source_xyz, np.ones((24, 1)), axis=1)

        ccm = np.linalg.pinv(source_xyz_hm).dot(reference_xyz)
        self.ccm = ccm
    def predict(self,input):
        input_img = Image.open(input, 'r').convert("RGB")
        input_img = self.deGamma(input_img, gamma=1.0)
        input_img = self.sRGB2XYZ(input_img)
        input_img = self.correctColor(input_img)

        input_img = self.XYZ2sRGB(input_img)
        input_img = self.applyGamma(input_img, gamma=1.0)
        return input_img
