import numpy as np
import csv
from scipy.linalg import pinv2, svd
import argparse
import numpy as np
from skimage import color,io
import csv
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics.pairwise import sigmoid_kernel,rbf_kernel
from colorCorrection.colorConverter import rgb2lab,rgbToLab,rgbToXyz
class KPLSRColorCorrection:
    def __init__(self):
        pass
    def trainComponent(self,X, Y):
        tol = 1e-06
        u = np.random.uniform(low=0.1, high=5, size=(24,))
        x_weights_old = 100
        for i in range(500):
            w = np.dot(X.T, u)
            t = np.dot(X, w)
            t /= np.sqrt(np.dot(t, t))
            c = np.dot(Y.T, t)
            #         c /= np.sqrt(np.dot(c,c)) + eps
            u = np.dot(Y, c) / (np.dot(c, c))
            u /= np.sqrt(np.dot(u, u))
            x_weights_diff = w - x_weights_old
            if np.dot(x_weights_diff, x_weights_diff) < tol:
                break
            x_weights_old = w
            n_iter = i + 1
        #     print(n_iter)
        return w, c

    def _svd_flip_1d(self,u, v):
        """Same as svd_flip but works on 1d arrays, and is inplace"""
        # svd_flip would force us to convert to 2d array and would also return 2d
        # arrays. We don't want that.
        biggest_abs_val_idx = np.argmax(np.abs(u))
        sign = np.sign(u[biggest_abs_val_idx])
        u *= sign
        v *= sign
    def center_scale_xy(self,X, Y, scale=True):
        x_mean = X.mean(axis=0)
        X -= x_mean
        y_mean = Y.mean(axis=0)
        Y -= y_mean
        # scale
        if scale:
            x_std = X.std(axis=0, ddof=1)
            x_std[x_std == 0.0] = 1.0
            X /= x_std
            y_std = Y.std(axis=0, ddof=1)
            y_std[y_std == 0.0] = 1.0
            Y /= y_std
        else:
            x_std = np.ones(X.shape[1])
            y_std = np.ones(Y.shape[1])
        return X, Y, x_mean, y_mean, x_std, y_std

    def select_kernel(self,source_raw,y=None):
        if y is None:
            if self.kernel=="sigmoid":
                return sigmoid_kernel(source_raw, gamma=self.z**(-2))
            elif self.kernel == "rbf":
                return rbf_kernel(source_raw, gamma=self.z**(-2))
        else:
            if self.kernel=="sigmoid":
                return sigmoid_kernel(source_raw,Y=y, gamma=self.z**(-2))
            elif self.kernel == "rbf":
                return rbf_kernel(source_raw,Y=y, gamma=self.z**(-2))
    def white_balance(self,src):
        x_r = 255 / src[18][0]
        x_g = 249 / src[18][1]
        x_b = 253 /src[18][2]
        x = [x_r,x_g,x_b]
        src *= x
        self.x = x
        return src
    def train(self,source,ref,kernel,z,n=4):
        self.kernel = kernel
        self.z = z
        self.ref = ref
        self.source = source
        reference_raw = np.loadtxt(ref, delimiter=",", skiprows=1, usecols=(1, 2, 3))
        reference_raw2 = np.loadtxt(ref, delimiter=",", skiprows=1, usecols=(1, 2, 3))
        source_raw = np.loadtxt(source, delimiter=",", skiprows=1, usecols=(1, 2, 3))
        source_raw2 = np.loadtxt(source, delimiter=",", skiprows=1, usecols=(1, 2, 3))
        
        reference_raw = self.white_balance(reference_raw)
        reference_raw2 = self.white_balance(reference_raw2)
        
        source_raw, reference_raw, x_mean, y_mean, x_std, y_std = (
            self.center_scale_xy(source_raw, reference_raw))
        self.x_mean = x_mean
        self.y_mean = y_mean
        self.x_std = x_std
        self.y_std = y_std
        source_raw2, reference_raw2, x_mean2, y_mean2, x_std2, y_std2 = (
            self.center_scale_xy(source_raw2, reference_raw2))
        source_raw = self.select_kernel(source_raw)
        source_raw2 = self.select_kernel(source_raw2)
        Y_eps = np.finfo(reference_raw.dtype).eps
        i = source_raw.shape[1]
        j = source_raw.shape[0]
        k = reference_raw.shape[1]
        x_weight = np.zeros((i, n))
        y_weight = np.zeros((k, n))
        x_score = np.zeros((j, n))
        y_score = np.zeros((j, n))
        x_loading = np.zeros((i, n))
        y_loading = np.zeros((k, n))
        for k in range(n):
            Yk_mask = np.all(np.abs(reference_raw) < 10 * Y_eps, axis=0)
            reference_raw[:, Yk_mask] = 0.0
            w, c = self.trainComponent(source_raw, reference_raw)
            x_weight[:, k] = w
            y_weight[:, k] = c
            self._svd_flip_1d(w, c)
            x_s = np.dot(source_raw, w)
            x_s /= np.sqrt(np.dot(x_s, x_s))
            y_s = np.dot(reference_raw, c)
            y_s /= np.sqrt(np.dot(y_s, y_s))
            x_score[:, k] = x_s
            y_score[:, k] = y_s
            x_l = np.dot(x_s, source_raw) / np.dot(x_s, x_s)
            a = (np.dot(np.dot(x_score, x_score.T), source_raw))
            source_raw -= a
            y_l = np.dot(x_s, reference_raw) / np.dot(x_s, x_s)
            reference_raw -= (np.dot(np.dot(x_score, x_score.T), reference_raw))
            x_loading[:, k] = x_l
            y_loading[:, k] = y_l

        temp_k = np.dot(source_raw2, source_raw2.T)

        temp = np.dot(x_score.T, temp_k)
        temp = np.dot(temp, y_score)

        B = np.dot(np.dot(source_raw2.T, y_score), pinv2(temp))
        B = np.dot(np.dot(B, x_score.T), reference_raw2)
        B = B * y_std
        self.B = B
        test = np.dot(source_raw2,B)
        test += self.y_mean
        print(test)
        self.x=[]
        for i in range(24):
            if i == 18:
                val_temp = test[i]
                x_r = 255 / val_temp[0]
                x_g = 249 / val_temp[1]
                x_b = 253 /val_temp[2]
                self.x = [x_r,x_g,x_b]
                break

    def predict(self,input):
        source_raw4 = np.loadtxt(self.source, delimiter=",", skiprows=1, usecols=(1, 2, 3))
        input_img = Image.open(input, 'r').convert("RGB")
        input_img = np.asarray(input_img)

        width, height, dim = input_img.shape
        input_img = input_img.reshape(width * height, dim)
        input_img = input_img - self.x_mean
        input_img = input_img / self.x_std
        source_raw4 -= self.x_mean
        source_raw4 /= self.x_std
        input_img = self.select_kernel(input_img,y=source_raw4)

        Y_pred = np.dot(input_img, self.B)
        Y_pred = Y_pred + self.y_mean
        Y_pred = Y_pred.reshape(width, height, dim)
        Y_pred2 = np.copy(Y_pred)
        Y_pred *= self.x
        Y_pred[Y_pred < 0] = 0
        Y_pred[Y_pred > 255] = 255
        Y_pred = Y_pred.astype(np.uint8)
        input_img = Image.fromarray(Y_pred, 'RGB')
        Y_pred2[Y_pred2 < 0] = 0
        Y_pred2[Y_pred2 > 255] = 255
        Y_pred2 = Y_pred2.astype(np.uint8)
        input_img2 = Image.fromarray(Y_pred2, 'RGB')
        return input_img,input_img2
