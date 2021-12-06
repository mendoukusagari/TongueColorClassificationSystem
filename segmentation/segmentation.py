from keras.models import *
import skimage.io as io
from utils import data_preprocessing
from . import masking
from skimage import img_as_uint
import time

class Segmentation():
    def segment_image(self,corrected,resized,out):
        
        start = time.time()
        json_file = open('./segmentation/model/model2.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("./segmentation/model/model2.h5")
        print("Loaded model from disk")

        loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        testGene = data_preprocessing.predictGenerator(resized)
        results = loaded_model.predict(testGene)
        results = results > 0.5
        io.imsave("./segmentation/tmp/msk.png", img_as_uint(results[0]))

        masking.mask(corrected,"./segmentation/tmp/msk.png",out)
        
        end = time.time()
        
        print(f"Runtime of segmentation is {end - start}")