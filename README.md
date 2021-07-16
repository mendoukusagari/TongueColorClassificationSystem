# TongueColorClassificationSystem

This is a system for classifying tongue based on the color. There are four steps in this system which are image acquisition, color correction, image segmentation, and color classification. The color correction provides 5 algorithms to be chosen: PCC, RPCC, PLSR, K-PLSR, and K-PLSRO, but this research is mainly using K-PLSRO as an improvement. The image segmentation is done by using U-Net Architecture. The color classification is done by using euclidean distance. 
For more details, you can check it out in the paper: https://github.com/dcmeta/TongueColorClassificationSystem/blob/master/TongueColorClassificationSystem.pdf

The result of this research can be seen below: 

Result of Correction:

![alt text](https://github.com/dcmeta/TongueColorClassificationSystem/blob/master/res_correction.png)

Result of Classification: 

![alt text](https://github.com/dcmeta/TongueColorClassificationSystem/blob/master/res_classification.png)
