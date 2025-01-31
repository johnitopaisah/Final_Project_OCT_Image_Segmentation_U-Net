# **Retinal OCT Segmentation using U-Net**

This repository contains the implementation of a U-Net-based model for retinal segmentation in Optical Coherence Tomography (OCT) images. The project focuses on accurately segmenting retinal regions to assist in diagnosing and monitoring retinal diseases.

---

## **Table of Contents**
1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Model Architecture](#model-architecture)  
4. [Training and Evaluation](#training-and-evaluation)  
5. [Installation](#installation)  
6. [Usage](#usage)  
7. [Results](#results)  
8. [Contributors](#contributors)  

---

## **Project Overview**
The goal of this project is to develop a robust and efficient U-Net model for retinal segmentation in OCT images. The model is designed to identify key retinal features and assist in medical image analysis.  

### **Features:**
- Implementation of U-Net in PyTorch.  
- Custom loss function combining binary cross-entropy and Dice coefficient.  
- Automated evaluation of segmentation performance using metrics like Dice Coefficient and Intersection over Union (IoU).  

---

## **Dataset**
This project uses a dataset of grayscale OCT images for training and testing. Each image is paired with its corresponding ground truth segmentation mask.  
- **Preprocessing**: Images are resized to 480 × 960 pixels and normalized.  
- **Augmentation**: Data augmentation techniques, including flipping, rotation, and contrast adjustments, are applied to improve model robustness.  

---

## **Model Architecture**
The model leverages the U-Net architecture, which is widely used for image segmentation tasks.  
### **Key Components**:
- **Encoder**: Extracts high-level features using convolutional and pooling layers.  
- **Bottleneck**: Captures the most critical features at the center of the network.  
- **Decoder**: Reconstructs the segmented output using transpose convolution and skip connections.  

![U-Net Architecture](./images/u-net-architecture.png) <!-- Add your own diagram if needed -->

---

## **Training and Evaluation**
### **Training**:
- **Optimizer**: Adam optimizer with a learning rate of 0.001.  
- **Loss Function**: Combination of Binary Cross-Entropy and Dice Loss for effective segmentation.  
- **Epochs**: 50  
- **Batch Size**: 16  

### **Evaluation Metrics**:
1. **Dice Coefficient**: Measures overlap between predicted and ground truth masks.  
2. **IoU (Jaccard Index)**: Assesses segmentation quality.  
3. **Precision, Recall, F1-Score**: For detailed performance evaluation.  

---

## **Installation**
1. Clone the repository:  
   ```bash
   git clone https://github.com/johnitopaisah/Final_Project_OCT_Image_Segmentation_U-Net.git
   cd Final_Project_OCT_Image_Segmentation_U-Net

2. Create a virtual environment and install all the dependencies
   ```bash
   python -m venv u-net     # you can call 'u-net' any name
   source venv/bin/activate # On Windows, use 'venv\Scripts\activate
   pip install -r requirements.txt

## **Usage**
1. **Preprocess the Dataset:**
Place OCT images and their segmentation masks in the data dir
2. **Train the Model:**
 ```bash
 python3 train.py
 ```

3. **Test on New Images:**
 ```bash
 python3 test.py
 ```

## **Result**
The model achieved the following performance metrics on the test dataset:

- **Mean Dice Coefficient:** `0.86 ± 0.05`
- **Mean Intersection over Union (IoU):** `0.80 ± 0.06`
