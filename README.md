# 3D-dilated-U-net
This study aimed to develop a 3D U-net-based, fully automatic masseter muscle segmentation on magnetic resonance images.

This is a self-contained repository for training classification-based, highly-customizable UNET models in Keras. It also lets you train on large datasets with augmentation and without having to load them all directly into memory. All you need is a CSV with file paths to the images.


## Model
![image](https://user-images.githubusercontent.com/6081278/172671643-482cbd27-58aa-429a-9915-b1b705e0c875.png)
Proposed 3D U-Net with dilated convolution network for 3D MRI masseter segmentation. Each blue box corresponds to a multi-channel feature map. The white boxes represent the copied feature map. The arrows indicate different operations. The numbers beside the blue boxes indicate the filter size, and the numbers next to D indicate the dilation rate.


## Results

![image](https://user-images.githubusercontent.com/6081278/172672292-c8d2b73c-c980-489b-b2bf-82fa4edce5ee.png)
(a)
![image](https://user-images.githubusercontent.com/6081278/172672305-cecc2da2-cdcb-485e-80c5-7958edd09bba.png)
(b)
![image](https://user-images.githubusercontent.com/6081278/172672312-0f01e3fc-5ae2-40ac-828c-03cbe93069d8.png)
(c)
![image](https://user-images.githubusercontent.com/6081278/172672828-9b75a5d5-22c2-4bd6-8594-34273e802824.png)
(d)
![image](https://user-images.githubusercontent.com/6081278/172672854-fcd28ca2-539f-47d2-bf1c-7950b0ffdb0c.png)
(e)
![image](https://user-images.githubusercontent.com/6081278/172672882-0bd6d233-50d1-4df5-930b-e06c9679819f.png)
(f)
![image](https://user-images.githubusercontent.com/6081278/172672916-be87350b-c7d3-43bb-8855-00c6890d3ce0.png)
(g)
![image](https://user-images.githubusercontent.com/6081278/172672972-c8ea8cb3-8f43-4e0d-a74b-ba1535545dec.png)
(h)

Fig. Example segmentation results: (a - c) Best segmentation of left masseter; (e - g) Best segmentation of right masseter; (d, h) 3D reconstruction. Red: reference; Blue: prediction; Purple: overlay of proposed segmentation versus ground truth.


![image](https://user-images.githubusercontent.com/6081278/172673460-ef7ea567-2279-46f1-9c13-348301d6ce8a.png)
  
Table.  DSC, sensitivity, specificity, precision, and root mean square distance (RMSD). The evaluations were performed over manual labels by an expert in dentistry. The agreement between comparisons is proportional to Dice, sensitivity, specificity, and precision and inversely proportional to HD and RMSD. Values are presented as mean ± SD. SD, standard deviation.
