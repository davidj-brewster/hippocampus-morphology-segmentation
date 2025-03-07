import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import measure, morphology, filters
import logging
import os

logging.basicConfig(level=logging.INFO)

def largest_connected_component(mask):
    """
    Finds the largest connected component in the mask.
    """
    labels = measure.label(mask)
    largest_cc_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
    return labels == largest_cc_label

def skull_stripping(image, output_path, min_size=500, area_threshold=5000):
    """
    Performs skull stripping on the given MRI image.
    
    Parameters:
    image (numpy.ndarray): MRI image to be processed.
    output_path (str): Path to save the skull stripping output.
    min_size (int): Minimum size of the objects to retain.
    area_threshold (int): Area threshold to remove small holes.
    
    Returns:
    brain_extracted (numpy.ndarray): Skull stripped image.
    brain_mask (numpy.ndarray): Mask of the brain region.
    """
    try:
        logging.info("Starting skull stripping")
        _, binary_mask = cv2.threshold((image * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_mask = morphology.remove_small_objects(binary_mask > 0, min_size=min_size)
        binary_mask = morphology.remove_small_holes(binary_mask, area_threshold=area_threshold)
        brain_mask = largest_connected_component(binary_mask)
        brain_extracted = image * brain_mask
        
        # Save visual representation of skull stripping
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(image, cmap='gray')
        
        plt.subplot(1, 2, 2)
        plt.title('Skull Stripped Image')
        plt.imshow(brain_extracted, cmap='gray')
        
        plt.savefig(output_path + "_skull_stripping.png")
        plt.close()

        logging.info("Skull stripping completed")
        return brain_extracted, brain_mask
    except Exception as e:
        logging.error(f"Error during skull stripping: {e}")
        raise

def edge_detection(image, threshold=0.2):
    """
    Performs edge detection on the given MRI image.
    
    Parameters:
    image (numpy.ndarray): MRI image to be processed.
    threshold (float): Threshold for edge detection.
    
    Returns:
    edges (numpy.ndarray): Edges detected in the image.
    """
    try:
        logging.info("Starting edge detection")
        edges = filters.sobel(image)
        edges = morphology.binary_dilation(edges > threshold)
        edge_coords = np.argwhere(edges)
        logging.info(f"Edge coordinates detected with threshold {threshold}: {len(edge_coords)} edges")
        return edges
    except Exception as e:
        logging.error(f"Error during edge detection: {e}")
        raise

def region_of_interest(edges, region, height, width):
    """
    Extracts the region of interest from the given edges.
    
    Parameters:
    edges (numpy.ndarray): Edges detected in the image.
    region (str): Region of interest to extract.
    height (int): Height of the image.
    width (int): Width of the image.
    
    Returns:
    roi (numpy.ndarray): Region of interest.
    """
    logging.info(f"Extracting region of interest: {region}")
    if region == 'left_hippocampus':
        return edges[height // 3: 2 * height // 3, width // 2 - width // 12: width // 2]
    elif region == 'right_hippocampus':
        return edges[height // 3: 2 * height // 3, width // 2: width // 2 + width // 12]
    elif region == 'temporal_lobe':
        return edges[7 * height // 8: height, width // 3: 2 * width // 3]
    else:
        raise ValueError("Invalid region specified")

def extract_region(brain_image, edges, region, height, width):
    """
    Extracts the specified region from the given edges and brain image.
    
    Parameters:
    brain_image (numpy.ndarray): Original MRI image.
    edges (numpy.ndarray): Edges detected in the image.
    region (str): Region to extract.
    height (int): Height of the image.
    width (int): Width of the image.
    
    Returns:
    region_mask (numpy.ndarray): Mask of the extracted region.
    """
    try:
        logging.info(f"Extracting region of interest: {region}")
        roi = region_of_interest(edges, region, height, width)
        logging.info(f"Region of interest '{region}' extracted with shape: {roi.shape}")

        labeled_roi = measure.label(roi)
        regions = measure.regionprops(labeled_roi)

        filtered_regions = [r for r in regions if 200 < r.area < 80000]

        region_mask = np.zeros((height, width), dtype=bool)
        if filtered_regions:
            largest_region = max(filtered_regions, key=lambda r: r.area)
            for coord in largest_region.coords:
                y, x = coord
                if region == 'left_hippocampus':
                    region_mask[y + height // 3, x + width // 2 - width // 12] = True
                elif region == 'right_hippocampus':
                    region_mask[y + height // 3, x + width // 2] = True

        selected_edge_coords = np.nonzero(region_mask)
        logging.info(f"Selected edge coordinates for {region}: {len(selected_edge_coords[0])} edges")

        logging.info(f"{region.capitalize()} extracted with area: {np.sum(region_mask)}")
        return region_mask
    except Exception as e:
        logging.error(f"Error extracting region of interest: {e}")
        raise

def detect_anomalies_in_hippocampus(brain_image, edges, skull_mask, height, width):
    """
    Detects anomalies in the hippocampal regions.
    
    Parameters:
    brain_image (numpy.ndarray): Original MRI image.
    edges (numpy.ndarray): Edges detected in the image.
    skull_mask (numpy.ndarray): Mask of the skull stripped image.
    height (int): Height of the image.
    width (int): Width of the image.
    
    Returns:
    left_mask (numpy.ndarray): Mask of the left hippocampal region with anomalies.
    right_mask (numpy.ndarray): Mask of the right hippocampal region with anomalies.
    """
    try:
        logging.info("Detecting anomalies in hippocampal regions")
        left_mask = extract_region(brain_image, edges, 'left_hippocampus', height, width)
        right_mask = extract_region(brain_image, edges, 'right_hippocampus', height, width)

        # Apply skull mask to the extracted regions
        left_mask &= skull_mask
        right_mask &= skull_mask

        logging.info("Detected anomalies in hippocampal regions")
        return left_mask, right_mask
    except Exception as e:
        logging.error(f"Error detecting anomalies in hippocampal regions: {e}")
        raise

def visualize_results(image, edges, left_mask, right_mask, output_path):
    """
    Visualizes the original image, skull stripped image, edges detected, and hippocampal regions with anomalies.
    
    Parameters:
    image (numpy.ndarray): Original MRI image.
    edges (numpy.ndarray): Edges detected in the image.
    left_mask (numpy.ndarray): Mask of the left hippocampal region.
    right_mask (numpy.ndarray): Mask of the right hippocampal region.
    output_path (str): Path to save the visualization output.
    """
    try:
        height, width = image.shape
        logging.info(f"Image dimensions: {height} x {width}")

        # Create the combined overlay with edges and anomalies
        combined_overlay = np.zeros((height, width, 3), dtype=np.float32)
        combined_overlay[..., 0] = image  # Red channel
        combined_overlay[..., 1] = image  # Green channel
        combined_overlay[..., 2] = image  # Blue channel

        # Mark the edges in blue
        combined_overlay[edges, 2] = 1.0  # Blue channel

        # Mark the hippocampal anomalies in red
        combined_overlay[left_mask, 0] = 1.0  # Red channel for left mask
        combined_overlay[right_mask, 0] = 1.0  # Red channel for right mask

        # Plot the images
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')

        axes[0, 1].imshow(edges, cmap='gray')
        axes[0, 1].set_title('Edges Detected')

        axes[1, 0].imshow(combined_overlay)
        axes[1, 0].set_title('Left hippocampus Edges and Anomalies')

        axes[1, 1].imshow(combined_overlay)
        axes[1, 1].set_title('Right hippocampus Edges and Anomalies')

        for ax in axes.flatten():
            ax.axis('off')

        plt.savefig(output_path + "_visualization.png")
        plt.close()

        logging.info(f"Visualization saved to {output_path}_visualization.png")
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        raise

def main(input_file, output_path, threshold=0.2, min_size=500, area_threshold=5000):
    """
    Main function to process the MRI image and detect anomalies.
    
    Parameters:
    input_file (str): Path to the input MRI image file.
    output_path (str): Path to save the output visualizations.
    threshold (float): Threshold for edge detection.
    min_size (int): Minimum size of the objects to retain during skull stripping.
    area_threshold (int): Area threshold to remove small holes during skull stripping.
    """
    try:
        # Load the image
        image = plt.imread(input_file)
        image = image.astype(np.float32) / 255.0  # Normalize the image

        # Perform skull stripping
        brain_image, skull_mask = skull_stripping(image, output_path, min_size=min_size, area_threshold=area_threshold)
        logging.info(f"Performed skull stripping on: {input_file}")

        # Perform edge detection
        edges = edge_detection(brain_image, threshold=threshold)
        logging.info(f"Performed edge detection on: {input_file}")

        # Get image dimensions
        height, width = brain_image.shape

        # Detect anomalies in the hippocampal regions
        left_mask, right_mask = detect_anomalies_in_hippocampus(brain_image, edges, skull_mask, height, width)

        # Visualize the results
        visualize_results(brain_image, edges, left_mask, right_mask, output_path)
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Brain MRI analysis")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input image file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output directory')
    
    args = parser.parse_args()
    main(args.input_file,args.output_path)


