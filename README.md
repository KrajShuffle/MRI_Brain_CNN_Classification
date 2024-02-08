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


