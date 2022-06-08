# 3D-dilated-U-net
This study aimed to develop a 3D U-net-based, fully automatic masseter muscle segmentation on magnetic resonance images.

This is a self-contained repository for training classification-based, highly-customizable UNET models in Keras. It also lets you train on large datasets with augmentation and without having to load them all directly into memory. All you need is a CSV with file paths to the images.

![image](https://user-images.githubusercontent.com/6081278/172671643-482cbd27-58aa-429a-9915-b1b705e0c875.png)
Proposed 3D U-Net with dilated convolution network for 3D MRI masseter segmentation. Each blue box corresponds to a multi-channel feature map. The white boxes represent the copied feature map. The arrows indicate different operations. The numbers beside the blue boxes indicate the filter size, and the numbers next to D indicate the dilation rate.
