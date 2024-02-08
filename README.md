# Convolutional Neural Network (CNN) for Brain Tumor Detection & Location Classification from MRI images via OpenCV & PyTorch

## Significance

According to research study conducted in 2015, based on 255 uninterrupted eight-hour workdays per year, radiologists are needing to review one image every three to four seconds to meet workload demands. However, the overall workload per radiologist is still increasing as MRI and CT scans are becoming more prevalent and the use of imaging contrasts/agents are increasing the complexity of information potentially gleaned from these scans. As a result, there are increasing cases of burnout amongst diagnostic radiologists. The aim of this project is to develop a deep learning approach that radiologists can potentially rely upon for to reduce their workloads. During this process, I hope to establish expertise in using PyTorch to develop deep learning models in a clean, and efficient manner with the hopes of being able to apply these skills to other similarly framed problems in the future. 

## Overall Dataset Characteristics
* 3 tumor-associated classes (Glioma, Pituitary, Meningioma) of mostly 512 x 512 sized images
* Non-tumor class has randomly sized images
* Scans are taken at different orientations: Axial, Coronal, and Sagittal

### Training Dataset & Class Breakdown
Composed of 5712 Images -> Split into ⅔ Training & ⅓ Validation

Glioma | Meningioma | Pituitary | No Tumor |
| :---: | :---: | :---: | :---:|
1321  | 1339 | 1457 | 1595 |

It is clear that there is a fairly even distribution amongst the respective classes, alleviating class imbalance concerns. Otherwise, I would need to incorporate techniques such as minority class penalties or a weighted random sampler to curate batches with evenly distributed classes.

### Test Dataset & Class Breakdown
Composed of 1311 images

Glioma | Meningioma | Pituitary | No Tumor |
| :---: | :---: | :---: | :---:|
300  | 306 | 300 | 405 |

## Model Objective
The goal is to accurately classify an MRI Brain image into one of these 4 classes, with the tumor classes varying based on brain location.

## Data Processing Workflow
1. Converting Image to Grayscale (Eliminates 2 image channels which were unnecessary & simplifies processing)

![MRI_to_GS](/Images/MRI_GraySc.png)
2. Apply a binary mask to separate brain from background and aid identification of Region of Interest (ROI) where contour is drawn

![ROI_ID](/Images/MRI_Mask_DrawCont.png)

3. Use coordinates of drawn contour to crop to ROI

![CMP_ROI_OG](/Images/ROI_Crop_Res.png)
Above figure provides cropped ROI image on the left and the original image on the right

4. Standardize images to fixed size of (256, 256) for training convolutional neural network

![Processed_OG](/Images/Standardize_Size.png)
On the left: final standardized, processed CNN input & On the right: original image if simply resized to (256,256)

**Image processing resulted in heavy elimination of background, amplifying area occupied by ROI in (256,256) frame** 

## Hyperparameter Optimization Results 
Optimizing number of layers, filters per layer, kernel sizes, & number of training epochs, batch size, learning rate, L2 Regularization

![HP_OP_Res](/Images/HPOP_Res.png)

This was not a true gridsearch as it is typically a pseudo outer product to yield all combinations of all realistic values chosen for each of the tunable hyperparameters. Due to time constraints, some of the hyperparameters (batch size, learning rate, L2 regularization) were established over fewer tests compared to other hyperparameters. However, further optimization of those hyperparameters would not yield as much of a performance gain in comparison to optimizing the other hyperparameters. Also, this is a sample of the entire hyperparameter optimization testing results, but highlights the final chosen values for the tunable hyperparameters. 

## Convolutional Neural Network Architecture

![Conv_Nw](/Images/ConvNetwork.png)

Used Batch Normalization & ReLU as the Activation Function throughout the Convolution Layers

## Classification Metrics on Test Dataset
Overall Accuracy: 90.5%

![TestDS_CM](/Images//TestDS_CM.png)

![TestDS_Metrics](/Images/TestDSQMetrics.png)



