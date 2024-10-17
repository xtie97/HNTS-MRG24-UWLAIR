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
   In the final testing phase: 
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

![Pre-RT Model Architecture](./images/pre_rt_model.jpg)

### Task 2: Mid-Radiotherapy Segmentation with Mask-Aware Attention

![Mid-RT Model Architecture](./images/mid_rt_model_updated.jpg)


## 💡 Example Use Cases

### Segmentation of GTVs in MRI Scans:
- **Example 1**: Pre-RT GTV segmentation
  ![Example 1](./images/example_pre_rt.jpg)
  
- **Example 2**: Mid-RT GTV segmentation
  ![Example 2](./images/example_mid_rt.jpg)

---

## 🛠 Installation

To install and run this project, you can use the pre-configured **Docker** container for easy setup. The Docker image is hosted on Docker Hub.

### Steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/xtie97/HNTS-MRG24-UWLAIR.git
   cd HNTS-MRG24-UWLAIR
   ```
2. **Pull the Docker image from Docker Hub**:
   ```bash
   docker pull xtie97/monai_wb
   ```

3. **Run the Docker container**:
   ```bash
   docker run -it --rm -v $(pwd):/workspace xtie97/monai_wb
   ```
- This command mounts your current project directory ($(pwd)) to the /workspace folder in the container for seamless access to your code.
- Use -it for interactive mode and --rm to remove the container once you are done.


## ⚙️ Usage
Once inside the Docker container, for Task1, please go to the folder **Task1_preRT**. Feel free to make changes in **configs/hyper_parameters.yaml**. Note that the parameters in this yaml file are the same with those used in our final submission. After specifying the parameters (especially the path to your data root and data list), you can start running the training.
```bash
cd Task1_preRT
python run.py
```

## Models:
We also release all the models in our final submission. Feel free to download them via the following link:


## 🙏 Acknowledgements
We acknowledge the organizers of the HNTS-MRG 24 Challenge for releasing high-quality, well-annotated data and for holding such a great challenge to advance the field of image-guided adaptive radiotherapy. We also thank the Center for High Throughput Computing (CHTC) at University of Wisconsin-Madison for providing GPU resources. 
