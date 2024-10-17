# 🎯 Deep Learning for Longitudinal Gross Tumor Volume Segmentation in MRI-Guided Adaptive Radiotherapy 🎯

![GitHub Stars](https://img.shields.io/github/stars/xtie97/HNTS-MRG24-UWLAIR?style=social) 
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

## 🚀 Overview
Accurate segmentation of gross tumor volume (GTV) is crucial for effective MRI-guided adaptive radiotherapy (MRgART) in head and neck cancer. Manual segmentation is labor-intensive and susceptible to interobserver variability. This project aims to address these challenges by leveraging **deep learning (DL)** models to automatically delineate GTVs on both **pre-radiotherapy (pre-RT)** and **mid-radiotherapy (mid-RT)** MRI scans. 

We present a series of DL models for longitudinal GTV segmentation, offering potential to streamline radiation oncology workflows in ART.

> **The code in this repository is provided to ensure the reproducibility of our submission to the [MICAAI HNTS-MRG 2024 grand challenge](https://hntsmrg24.grand-challenge.org/overview/).** 


## Key Features
- **Pre-Radiotherapy (Pre-RT) GTV Segmentation (Task 1)**:  
  The DL models trained on combined pre-RT and mid-RT MRI datasets, yielding improved accuracy on hold-out test sets compared to models trained solely on pre-RT data.
  
- **Mid-Radiotherapy (Mid-RT) GTV Segmentation (Task 2)**:  
  The DL models integrating prior information from pre-RT scans can significantly improve the performance. Introduced **mask-aware attention modules** to leverage pre-RT GTV masks in mid-RT segmentation. 

- **Ensemble Approach**:  
  Utilized an ensemble of 10 models for both tasks to improve robustness and overall performance.

## 📊 Results and Performance
- **Pre-Radiotherapy (Task 1)**:
  - Average **DSCagg**: 0.794
  - **Primary GTV (GTVp)** DSC: 0.745
  - **Metastatic Lymph Nodes (GTVn)** DSC: 0.844

- **Mid-Radiotherapy (Task 2)**:
  - Average **DSCagg**: xxx
  - **GTVp** DSC: xxx
  - **GTVn** DSC: xxx
    
## 🧠 Model Architectures and Visualizations

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
```

## ⚙️ Usage
After installing the dependencies, use the following script to run GTV segmentation on MRI scans:
```bash
python run_segmenter.py --input <path_to_mri> --output <path_to_output>
```

## 🙏 Acknowledgements
This project was developed as part of the MICAAI Challenge. Special thanks to the research team for their guidance and collaboration.
