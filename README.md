# Wind Speed Time Series Forecasting with LSTM and TCN Models Using Physics-Informed Vectors

This repository implements a state-of-the-art machine learning framework that integrates **Physics-Informed Vectors (PIV)** with **Long Short-Term Memory (LSTM)** and **Temporal Convolutional Networks (TCN)** for improved wind speed forecasting. The project enhances predictive accuracy by incorporating domain-specific physical knowledge and custom hybrid loss functions.

## Overview of the Proposed Methodology

Below is a block diagram that outlines the framework, including data preprocessing, Physics-Informed Vector computation, and model training using LSTM and TCN architectures:

![Block Diagram](block.png)


## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Running the Project](#running-the-project)
- [Custom Hybrid Loss Function](#Custom-Hybrid-Loss-Function)
- [Results](#results)
- [Publication](#Publication)
- [License](#license)

## Installation

To get started with the project, follow the steps below:

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-name>

### Set up a virtual environment

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

pip install -r requirements.txt
## Project Structure

The following is an overview of the repository's folder structure. The dataset folders (`20_Percent`, `50_Percent`, `75_Percent`, and `All`) contain data from different geographic locations, divided by percentage of available data for performance evaluation.


```
.
├── 20_Percent
│   ├── canada_data
│   │   ├── ...
│   ├── chilli_data
│   │   ├── ...
│   ├── kazakistan
│   │   ├── ...
│   └── Mongolia
│       ├── ...
├── 50_Percent
│   ├── canada_data
│   │   ├── ...
│   ├── chilli_data
│   │   ├── ...
│   ├── kazakistan
│   │   ├── ...
│   └── Mongolia
│       ├── ...
├── 75_Percent
│   ├── canada_data
│   │   ├── ...
│   ├── chilli_data
│   │   ├── ...
│   ├── kazakistan
│   │   ├── ...
│   └── Mongolia
│       ├── ...
└── All
    ├── canada_data
    │   ├── ...
    ├── chilli_data
    │   ├── ...
    ├── kazakistan
    │   ├── ...
    └── Mongolia
        ├── ...
```

## Running the Project
1. Navigate to the Desired Dataset Folder: Change directory to the specific dataset folder (e.g., 20_Percent, 50_Percent, or 75_Percent).
2. Modify Shell Script Parameters (if necessary): Open the corresponding .sh file (e.g., run.sh) and adjust parameters such as NUM_EPOCHS, BATCH_SIZE, and INITIAL_LR as needed.
3. Execute the Shell Script: Run the script in your terminal:
bash run.sh
## Custom Hybrid Loss Function

Our model employs a **custom hybrid loss function** that incorporates **Physics-Informed Vectors (PIV)** and traditional Mean Squared Error (MSE). The PIV term initially has a strong influence on predictions, guiding the model towards physical accuracy, but this influence gradually decays during training.

The loss function is defined as follows:

$$L = \frac{1}{N} \sum_{i=1}^{N} (S_{t+1,i} - \hat{\hat{S}}_{t+1,i})^2 + \lambda_0 e^{-pt} \frac{1}{N} \sum_{i=1}^{N} (\hat{\hat{S}}_{t+1,i} - \hat{S}_{t+1,i})^2$$


Where:
- \( \hat{\hat{S}}_{t+1,i} \): Model's predicted wind speed.
- \( \hat{S}_{t+1,i} \): Physics-informed estimated wind speed.
- \( N \): Number of observations.
- \( \lambda_0 e^{-pt} \): Decaying term that reduces the PIV influence as training progresses, allowing the model to focus on minimizing prediction errors.

This hybrid loss function improves model convergence and predictive accuracy by initially aligning the model closely with physics-based predictions and later tuning it to focus on data-driven corrections.

## Results

The integration of **Physics-Informed Vectors (PIV)** significantly improves predictive performance across multiple datasets. Below are the performance metrics across different subsets of data:

### Key Performance Metrics

- **Mean Absolute Error (MAE):**  
  ![MAE Improvement](./MAE.png)
- **Mean Squared Error (MSE):**  
  ![MSE Improvement](./MSE.png)
- **R-Squared (R²):**  
  ![R2 Improvement](./R2.png)

The graphs above illustrate the improvements achieved by incorporating PIV into standard LSTM and TCN models. The percentage improvements are noted in terms of accuracy and convergence speed.


## Citation and Publication

This work has been presented and published at the **2024 14th Asian Control Conference (ASCC)**.

**Title:** Integrating Physics-Informed Vectors for Improved Wind Speed Forecasting with Neural Networks  
**Authors:** Laeeq Aslam, Runmin Zou, Ebrahim Awan, Sharjeel Abid Butt  
**Conference:** 2024 14th Asian Control Conference (ASCC)  
**Pages:** 1902-1907  
**Year:** 2024  
**Organization:** IEEE  
**[Link to Paper](https://ieeexplore.ieee.org/abstract/document/10665742)**

### Citation:
```bibtex
@inproceedings{aslam2024integrating,
  title={Integrating Physics-Informed Vectors for Improved Wind Speed Forecasting with Neural Networks},
  author={Aslam, Laeeq and Zou, Runmin and Awan, Ebrahim and Butt, Sharjeel Abid},
  booktitle={2024 14th Asian Control Conference (ASCC)},
  pages={1902--1907},
  year={2024},
  organization={IEEE}
}



