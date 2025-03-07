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

## Features and Limitations

### Functionality

Preprocessing: The preprocessing step involves normalizing the image, which is essential for standardizing the input data.
Skull Stripping: This step removes non-brain tissues, focusing the analysis on the brain regions. The use of thresholding and morphological operations is appropriate for this task.
Edge Detection: The code supports multiple edge detection methods (Canny and Gabor), which is a good approach to compare and choose the best method for hippocampus segmentation.
Anomaly Detection: Functions like detect_bright_spots, detect_dark_spots, detect_density_anomalies, and detect_outliers are included to identify anomalies in the hippocampal regions. This adds robustness to the analysis.
Visualization: The visualization functions are well-designed to display the results of the segmentation and anomaly detection processes. This is crucial for verifying the correctness of the results.

### Approach and Algorithms

Frequency Analysis: The use of frequency analysis provides an additional perspective on the image data, which can be useful for certain types of analysis.
Morphological Operations: These operations are used extensively for cleaning up the masks and refining the segmentation results. This is a standard approach in image processing.
Intensity-Based Refinement: The refinement of hippocampal masks based on intensity values helps in improving the accuracy of the segmentation.

