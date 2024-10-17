# 🎯 Deep Learning for GTV Segmentation in MRI-Guided Adaptive Radiotherapy (MRgART) 🎯

![Project Banner](./images/banner.png)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) 
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue) 
![GitHub Stars](https://img.shields.io/github/stars/xtie97/HNTS-MRG24-UWLAIR?style=social) 
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

## Overview
Accurate segmentation of gross tumor volume (GTV) is crucial for effective MRI-guided adaptive radiotherapy (MRgART) in head and neck cancer. Manual segmentation is labor-intensive and susceptible to interobserver variability. This project aims to address these challenges by leveraging **deep learning (DL)** models to automatically delineate GTVs on both **pre-radiotherapy (pre-RT)** and **mid-radiotherapy (mid-RT)** MRI scans. 

We present a series of DL models for longitudinal GTV segmentation, offering potential to streamline radiation oncology workflows in ART.

The code in this repository is to ensure reproducibility of our submission to MICAAI HNTS-MRG 2024 grand challenge. 

## Key Features
- **Pre-Radiotherapy (Pre-RT) Tumor Segmentation (Task 1)**:  
  A DL model trained on combined pre-RT and mid-RT MRI datasets, yielding improved accuracy on hold-out test sets compared to models trained solely on pre-RT data.
  
- **Mid-Radiotherapy (Mid-RT) Tumor Segmentation (Task 2)**:  
  Introduced **mask-aware attention modules** to leverage pre-RT GTV masks in mid-RT data segmentation, providing slight performance gains over baseline methods.

- **Ensemble Approach**:  
  Utilized an ensemble of 10 models for both tasks to improve robustness and overall performance.

## Results
- **Task 1 (Pre-RT Segmentation)**:
  - Average **DSCagg**: 0.794
  - Primary GTV (GTVp) DSC: 0.745
  - Metastatic Lymph Nodes (GTVn) DSC: 0.844

- **Task 2 (Mid-RT Segmentation)**:
  - Average **DSCagg**: 0.733
  - GTVp DSC: 0.607
  - GTVn DSC: 0.859

## Model Architecture
- **Backbone**:  
  The backbone of the model is **SegResNet** with deep supervision, enabling better feature extraction and improved accuracy.

- **Mask-Aware Attention Modules (Task 2)**:  
  Used attention mechanisms to allow pre-RT GTV masks to influence intermediate features, enhancing segmentation accuracy for mid-RT data.

## Installation
To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/xtie97/HNTS-MRG24-UWLAIR.git
cd HNTS-MRG24-UWLAIR
pip install -r requirements.txt
