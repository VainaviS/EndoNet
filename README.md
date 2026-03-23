# EndoNet
EndoNet Web Interface for Automated Endometriosis Analysis

# EndoNet: Deep Learning Framework for Detection and Segmentation of Endometriosis Lesions

EndoNet is a deep learning framework for automated **detection and segmentation of endometriosis lesions** in laparoscopic images.
The framework has object detection and instance segmentation models to assist in identifying lesion locations and boundaries from surgical imagery.

This repository provides the complete training pipeline, preprocessing scripts, and a demo web interface for reproducible experimentation.

---

## Overview

Endometriosis is a chronic gynecological condition affecting approximately **1 in 10 women of reproductive age**, often requiring laparoscopic surgery for diagnosis. Manual visual inspection of lesions can be time-consuming and subjective.

EndoNet aims to assist clinical analysis by applying computer vision models to laparoscopic images for automated lesion identification.

The pipeline consists of two independent deep learning models:

* YOLO-based object detection for lesion localization
* Mask R-CNN instance segmentation for lesion boundary extraction

---

## Pipeline

The EndoNet workflow is structured as follows:

Input Laparoscopic Image-> YOLO Detection (Lesion Localization) -> Mask R-CNN Segmentation (Lesion Boundary Extraction)->
Visualization via Web Interface

The detection and segmentation models are trained separately and evaluated individually to analyze performance on laparoscopic datasets.

---

## Dataset

Training and evaluation were conducted using the **GLENDA dataset (Gynecologic Laparoscopy Endometriosis Dataset)**.

The dataset contains annotated laparoscopic frames with lesion masks categorized into four anatomical classes:

* Peritoneum
* Ovary
* TIE (Deep Infiltrating Endometriosis)
* Uterus

Dataset link:
https://ftp.itec.aau.at/datasets/GLENDA/

Due to licensing restrictions, the dataset is **not included in this repository**. Please download it from the official source and place it in the expected directory structure described below.

---

## Repository Structure

EndoNet/
│
├── notebooks
│   ├── yolo_training.ipynb
│   └── mask_rcnn_training.ipynb
│
├── preprocessing
│   ├── coco_to_yolo.py
│   └── dataset_split.py
│
├── webapp
│   ├── app.py
│   └── requirements.txt
│
├── sample_images
│   ├── img1
|   └── img2
│   └── img3
|
|
├── image outputs
│   ├── Image_outputs_maskRCNN(images from the Mask-RCNN model)
│   └── Image_outputs_yolo(images from the yolo model)
│
└── README.md

---
Models link- 
Yolo- https://drive.google.com/file/d/1cudHJCf45DAL7_A3BvDqP0maFZG8puwk/view?usp=sharing
MaskRCNN- https://drive.google.com/file/d/1yUea6N8ufVyUSnZKRBwvPTVTQ3krlO_f/view?usp=sharing 
## Installation

Clone the repository:

git clone https://github.com/yourusername/EndoNet.git
cd EndoNet

Install dependencies:

pip install -r webapp/requirements.txt

---

## Dataset Preparation

1. Download the GLENDA dataset.
2. Place images and annotations in a dataset directory.
3. Run preprocessing scripts to convert annotations and split the dataset.

Example:

python preprocessing/coco_to_yolo.py
python preprocessing/dataset_split.py

---

## Model Training

Training notebooks are provided in the `notebooks` directory.

YOLO Detection Model:

notebooks/yolo_training.ipynb

Mask R-CNN Segmentation Model:

notebooks/mask_rcnn_training.ipynb

These notebooks contain the complete training pipeline including dataset preparation, model configuration, and evaluation.

---

## Running the Web Interface

The repository includes a simple demo interface for visualizing model predictions.

Run the Streamlit app:

cd webapp
streamlit run app.py

The interface allows users to:

* Upload laparoscopic images
* Run lesion detection
* Visualize segmentation masks

---

## Example Output

The system produces two types of predictions:

Detection Output
Bounding boxes indicating lesion locations.

Segmentation Output
Pixel-wise masks highlighting lesion boundaries.

---

## Limitations

* Dataset size is relatively small for deep learning training.
* Class imbalance exists between lesion categories.
* The framework is intended for research purposes and not for clinical diagnosis.

---

## Future Work

Potential improvements include:

* Integration of detection and segmentation stages into a unified pipeline
* Expansion to larger laparoscopic datasets
* Clinical validation with surgical experts
