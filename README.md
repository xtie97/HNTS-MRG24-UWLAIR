# 🎯 Deep Learning for GTV Segmentation in MRI-Guided Adaptive Radiotherapy (MRgART) 🎯

![GitHub Stars](https://img.shields.io/github/stars/xtie97/HNTS-MRG24-UWLAIR?style=social) 
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

## Overview
Accurate segmentation of gross tumor volume (GTV) is crucial for effective MRI-guided adaptive radiotherapy (MRgART) in head and neck cancer. Manual segmentation is labor-intensive and susceptible to interobserver variability. This project aims to address these challenges by leveraging **deep learning (DL)** models to automatically delineate GTVs on both **pre-radiotherapy (pre-RT)** and **mid-radiotherapy (mid-RT)** MRI scans. 

We present a series of DL models for longitudinal GTV segmentation, offering potential to streamline radiation oncology workflows in ART.

The code in this repository is to ensure reproducibility of our submission to MICAAI HNTS-MRG 2024 grand challenge. 

## 📊 Results and Performance
- **Pre-Radiotherapy (Task 1)**:
  - Average **DSCagg**: 0.794
  - **Primary GTV (GTVp)** DSC: 0.745
  - **Metastatic Lymph Nodes (GTVn)** DSC: 0.844

- **Mid-Radiotherapy (Task 2)**:
  - Average **DSCagg**: 0.733
  - **GTVp** DSC: 0.607
  - **GTVn** DSC: 0.859

---

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

## 🧠 Model Architectures and Visualizations
Here is a brief overview of the models we developed for GTV segmentation, using **SegResNet** as the backbone with advanced supervision techniques and mask-aware attention.

### Task 1: Pre-Radiotherapy Segmentation

![Pre-RT Model Architecture](./images/pre_rt_model.png)

### Task 2: Mid-Radiotherapy Segmentation with Mask-Aware Attention

![Mid-RT Model Architecture](./images/mid_rt_model.png)

---

## 💡 Example Use Cases

### Segmentation of GTVs in MRI Scans:
- **Example 1**: Pre-RT GTV segmentation
  ![Example 1](./images/example_pre_rt.png)
  
- **Example 2**: Mid-RT GTV segmentation
  ![Example 2](./images/example_mid_rt.png)


## 🛠 Installation

To install and run this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/xtie97/HNTS-MRG24-UWLAIR.git
cd HNTS-MRG24-UWLAIR
pip install -r requirements.txt

## ⚙️ Usage
After installing the dependencies, use the following script to run GTV segmentation on MRI scans:
```bash
python run_segmenter.py --input <path_to_mri> --output <path_to_output>

## 🙏 Acknowledgements
This project was developed as part of the MICAAI Challenge. Special thanks to the research team for their guidance and collaboration.
