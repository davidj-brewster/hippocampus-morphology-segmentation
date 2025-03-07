# Hippocampus Morphology Segmentation

## Description
This project is an early attempt at segmenting and detecting the hippocampus using Python. The process involves preprocessing, skull-stripping, and applying medpy and skimage morphology methods to achieve segmentation results.

## Approach
The approach taken in this project can be broken down into several key steps:
1. **Preprocessing**: Preparing the raw MRI images for analysis, which includes resizing, normalizing intensity values, and noise reduction.
2. **Skull Stripping**: Removing non-brain tissues from MRI images to focus the analysis on the brain and its structures.
3. **Segmentation**: Partitioning the brain images into different regions, specifically isolating the hippocampus using medpy and skimage morphology methods.
4. **Anomaly Detection**: Identifying anomalies in the hippocampus regions through various image processing techniques.

## Libraries Used
- **numpy**: For numerical operations.
- **matplotlib**: For plotting and visualizing images.
- **scikit-image**: For image processing tasks.
- **opencv-python**: For computer vision tasks.
- **pillow**: For image loading and manipulation.

## Installation
To install the required dependencies, run the following command:
```sh
pip install -r requirements.txt

## Usage

I used the MSD brain segmentation publicly available dataset for training & inference - many thanks and kudos to those behind this comprehensive dataset.
Data format should align with this dataset
