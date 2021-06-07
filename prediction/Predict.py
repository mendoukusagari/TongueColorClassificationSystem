import numpy as np
from PIL import Image
from skimage import io, color
from skimage import io, color
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import time
class Predict():
    def __init__(self):
        cyan = np.array([76.0693, -0.5580, 1.3615])
        red = np.array([52.2540, 34.8412, 21.3002])
        purple = np.array([69.4695, 42.4732, -23.8880])
        dr = np.array([37.8424, 24.5503, 25.9396])
        lr = np.array([69.4695, 28.4947, 13.3940])
        lp = np.array([76.0693, 24.3246, -9.7749])
        bk = np.array([37.8424, 3.9632, 20.5874])
        gy = np.array([61.6542, 5.7160, 3.7317])
        w = np.array([70.9763, 10.9843, 8.2952])
        ye = np.array([56.681, 9.5539, 24.4546])
        X = [cyan, red, purple, dr, lr, bk, gy, w, ye]
        y = ['cyan', 'red', 'purple', 'dr', 'lr', 'bk', 'gy', 'w', 'ye']
        neigh = KNeighborsClassifier(n_neighbors=1, weights='distance', metric="euclidean")
        neigh.fit(X, y)
        self.neigh = neigh
        self.res = {}
    def averagePixel(self,input):
        input_img = Image.open(input, 'r').convert("RGB")
        input_img = np.asarray(input_img)
        input_img = color.rgb2lab(input_img)
        
        height, width, dim = input_img.shape
        input_img = input_img.reshape(width * height, dim)
        total = width*height
        clr = np.array([0.0, 0.0, 0.0])
        count = 0
        
        temp = {'cyan': [], 'red': [], 'purple': [], 'dr': [], 'lr': [], 'bk': [], 'gy': [], 'w': [],
                'ye': []}
        for key, value in temp.items():
            temp[key] = np.zeros((height, width, 3))
        lab = []
        x=0
        print(width*height)
        for i in input_img:
            if abs(i[0]) > 1 and abs(i[1]) > 1 and abs(i[2]) > 1:
#                clr += i
                count += 1
                lab.append([i[0],i[1],i[2]])
                res = self.predictPixel([i[0],i[1],i[2]])
                if res in self.res.keys():
                    self.res[res] += 1
                else:
                    self.res[res] = 0
                temp[res][int(x / width)][x-(width*int(x/width))] = [i[0],i[1],i[2]]
            x+=1
        for x in self.res.keys():
               self.res[x] = self.res[x] / count
               io.imsave('./tmp/ext/{}.png'.format(x), color.lab2rgb(temp[x]))
        lab = np.array(lab)
        l = np.median(lab[:,0])
        a = np.median(lab[:,1])
        b = np.median(lab[:,2])
#        avgpixel = clr / count
#        return avgpixel
        return self.res, [l,a,b]
    def predictPixel(self,avgpixel):
        res = self.neigh.predict([avgpixel])
        return res[0]

    def predict(self,input):
        start = time.time()
        avgPixel,lab = self.averagePixel(input)
        end = time.time()
        print(f"Runtime of classification is {end - start}")
        return avgPixel,lab,self.predictPixel(lab)
