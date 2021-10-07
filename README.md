# TongueColorClassificationSystem

This is a system for classifying tongue color under different sensor. There are four steps in this system which are image acquisition, color correction, image segmentation, and color classification. The color correction provides 5 algorithms to be chosen: PCC, RPCC, PLSR, K-PLSR, and K-PLSRO, but this research is mainly using K-PLSRO as an improvement. The image segmentation is done by using deep learning with U-Net Architecture. And the color classification is done by using euclidean distance. 
For more details, you can check it out in the paper: https://www.researchgate.net/publication/354686824_Sensor-Independent_Framework_for_Tongue_Color_Classification_System_TCCS

The result of this research can be seen below: 

Result of Correction:

![alt text](https://github.com/dcmeta/TongueColorClassificationSystem/blob/master/res_correction.png)

Result of Classification: 

![alt text](https://github.com/dcmeta/TongueColorClassificationSystem/blob/master/res_classification.png)

The result shows that the images which are corrected by using K-PLSRO can make a consistent classification in different sensor.
