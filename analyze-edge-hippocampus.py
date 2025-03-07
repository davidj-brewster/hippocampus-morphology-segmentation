from math import floor 
import unittest
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import filters, measure, morphology
from medpy.filter.binary import largest_connected_component
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_image(file_path):
    try:
        logging.info(f"Loading image from {file_path}")
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image from {file_path}")
        logging.info("Image loaded successfully")
        return image
    except Exception as e:
        logging.error(f"Error loading image: {e}")
        raise

def preprocess_image(image):
    try:
        logging.info("Starting image preprocessing")
        # Normalization
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        logging.info("Image normalization completed")
        return image
    except Exception as e:
        logging.error(f"Error during image preprocessing: {e}")
        raise

def skull_stripping(image):
    try:
        logging.info("Starting skull stripping")
        # Use a threshold to create a binary mask
        #_, binary_mask = cv2.threshold((image * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, binary_mask = cv2.threshold((image * 255).astype(np.uint8), 0, 255, (cv2.THRESH_OTSU + cv2.THRESH_BINARY))
        binary_mask = morphology.remove_small_objects(binary_mask > 0, min_size=5)  # Reduced threshold
        binary_mask = morphology.remove_small_holes(binary_mask, area_threshold=5)  # Reduced threshold
        brain_mask = largest_connected_component(binary_mask)
        brain_extracted = image * brain_mask
        logging.info("Skull stripping completed")
        return brain_extracted
    except Exception as e:
        logging.error(f"Error during skull stripping: {e}")
        raise


def edge_detection(image, output_path='/tmp/', threshold=0.05):
    try:
        logging.info("Starting edge detection")
        edges = filters.sobel(image)
        edges = morphology.binary_dilation(edges > threshold)
        edge_coords = np.argwhere(edges)
        logging.info(f"Edge coordinates detected with threshold {threshold}: {len(edge_coords)} edges")
        
        # Save visual representation of edges
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Skull Stripped Image')
        plt.imshow(image, cmap='gray')
        
        plt.subplot(1, 2, 2)
        plt.title('Edges Detected')
        plt.imshow(edges, cmap='gray')
        
        plt.savefig(output_path + "_edge_detection.png")
        plt.close()

        return edges
    except Exception as e:
        logging.error(f"Error during edge detection: {e}")
        raise

def v1_edge_detection(image, threshold=0.8):
    try:
        logging.info("Starting edge detection using local maxima")
        
        # Optionally, smooth the image
        smoothed_image = filters.gaussian(image, sigma=1)
        
        # Find local maxima
        local_maxima = morphology.max_tree_local_maxima(smoothed_image)
        
        edge_coords = np.argwhere(local_maxima)
        logging.info(f"Local maxima coordinates detected: {len(edge_coords)} maxima")
        
        return local_maxima
    except Exception as e:
        logging.error(f"Error during edge detection: {e}")
        raise

def extract_hippocampal_region(brain_image, edges, side='left'):
    try:
        logging.info(f"Extracting {side} hippocampal region")
        height, width = brain_image.shape

        total_edges = np.argwhere(edges)
        logging.info(f"Total edges detected in the whole image: {len(total_edges)} edges")

        # Define the ROI for the hippocampus focusing on the middle third both vertically and horizontally
        if side == 'left':
            roi = edges[height // 4: 3 * height // 4, width // 3: width // 2]
        elif side == 'right':
            roi = edges[height // 4: 3 * height // 4, width // 2: 2 * width // 2]
        else:
            raise ValueError("side must be 'left' or 'right'")

        roi_edge_coords = np.argwhere(roi)
        logging.info(f"Edges coordinates considered for {side} hippocampus: {len(roi_edge_coords)} edges")

        # Label connected components
        labeled_roi = measure.label(roi)
        regions = measure.regionprops(labeled_roi)

        # Filter regions based on anatomical location and size
        filtered_regions = [r for r in regions if 200 < r.area < 80000]

        half_sized_mask = np.zeros((height, width // 2), dtype=np.bool_)
        if filtered_regions:
            largest_region = max(filtered_regions, key=lambda r: r.area)
            if side == 'left':
                half_sized_mask[height // 4: 3*height // 4, 2 * width // 3:] = (roi == largest_region.label)
            elif side == 'right':
                half_sized_mask[height // 4: 3 * height // 4, :width // 3] = (roi == largest_region.label)

        selected_edge_coords = np.argwhere(half_sized_mask)
        logging.info(f"Selected edge coordinates for {side} hippocampus: {len(selected_edge_coords)} edges")

        logging.info(f"{side.capitalize()} hippocampal region extracted with area: {np.sum(half_sized_mask)}")
        return half_sized_mask
    except Exception as e:
        logging.error(f"Error extracting hippocampal region: {e}")
        raise

def v88_extract_hippocampal_region(brain_image, edges, side='left'):
    try:
        logging.info(f"Extracting {side} hippocampal region")
        height, width = brain_image.shape

        total_edges = np.argwhere(edges)
        logging.info(f"Total edges detected in the whole image: {len(total_edges)} edges")

        # Define the ROI for the hippocampus focusing on the middle third both vertically and horizontally
        if side == 'left':
            roi = edges[height // 3: 2 * height // 3, width // 3: width // 2]
        elif side == 'right':
            roi = edges[height // 3: 2 * height // 3, width // 2: 2 * width // 3]
        else:
            raise ValueError("side must be 'left' or 'right'")

        roi_edge_coords = np.argwhere(roi)
        logging.info(f"Edges coordinates considered for {side} hippocampus: {len(roi_edge_coords)} edges")

        # Label connected components
        labeled_roi = measure.label(roi)
        regions = measure.regionprops(labeled_roi)

        # Filter regions based on anatomical location and size
        filtered_regions = [r for r in regions if 2000 < r.area < 8000]

        half_sized_mask = np.zeros((height, width // 2), dtype=np.bool_)
        if filtered_regions:
            largest_region = max(filtered_regions, key=lambda r: r.area)
            if side == 'left':
                half_sized_mask[height // 3: 2 * height // 3, :width // 6] = (roi == largest_region.label)
            elif side == 'right':
                half_sized_mask[height // 3: 2 * height // 3, :width // 6] = (roi == largest_region.label)

        selected_edge_coords = np.argwhere(half_sized_mask)
        logging.info(f"Selected edge coordinates for {side} hippocampus: {len(selected_edge_coords)} edges")

        logging.info(f"{side.capitalize()} hippocampal region extracted with area: {np.sum(half_sized_mask)}")
        return half_sized_mask
    except Exception as e:
        logging.error(f"Error extracting hippocampal region: {e}")
        raise

def v84_extract_hippocampal_region(brain_image, edges, side='left'):
    try:
        logging.info(f"Extracting {side} hippocampal region")
        height, width = brain_image.shape

        total_edges = np.argwhere(edges)
        logging.info(f"Total edges detected in the whole image: {len(total_edges)} edges")

        # Define the ROI for the hippocampus focusing on the middle third both vertically and horizontally
        if side == 'left':
            roi = edges[height // 3: 2 * height // 3, width // 3: width // 2]
        elif side == 'right':
            roi = edges[height // 3: 2 * height // 3, width // 2: 2 * width // 3]
        else:
            raise ValueError("side must be 'left' or 'right'")

        roi_edge_coords = np.argwhere(roi)
        logging.info(f"Edges coordinates considered for {side} hippocampus: {len(roi_edge_coords)} edges")

        # Label connected components
        labeled_roi = measure.label(roi)
        regions = measure.regionprops(labeled_roi)

        # Filter regions based on anatomical location and size
        filtered_regions = [r for r in regions if 200 < r.area < 80000]

        half_sized_mask = np.zeros((height, width // 2), dtype=np.bool_)
        if filtered_regions:
            largest_region = max(filtered_regions, key=lambda r: r.area)
            if side == 'left':
                half_sized_mask[height // 3: 2 * height // 3, :width // 6] = (labeled_roi == largest_region.label)
            elif side == 'right':
                half_sized_mask[height // 3: 2 * height // 3, width // 6:] = (labeled_roi == largest_region.label)

        selected_edge_coords = np.argwhere(half_sized_mask)
        logging.info(f"Selected edge coordinates for {side} hippocampus: {len(selected_edge_coords)} edges")

        logging.info(f"{side.capitalize()} hippocampal region extracted with area: {np.sum(half_sized_mask)}")
        return half_sized_mask
    except Exception as e:
        logging.error(f"Error extracting hippocampal region: {e}")
        raise

def v82_extract_hippocampal_region(brain_image, edges, side='left'):
    try:
        logging.info(f"Extracting {side} hippocampal region")
        height, width = brain_image.shape

        total_edges = np.argwhere(edges)
        logging.info(f"Total edges detected in the whole image: {len(total_edges)} edges")

        # Define the ROI for the hippocampus focusing on the middle third both vertically and horizontally
        if side == 'left':
            roi = edges[height // 3: 2 * height // 3, width // 3: width // 2]
        elif side == 'right':
            roi = edges[height // 3: 2 * height // 3, width // 2: 2 * width // 3]
        else:
            raise ValueError("side must be 'left' or 'right'")

        roi_edge_coords = np.argwhere(roi)
        logging.info(f"Edges coordinates considered for {side} hippocampus: {len(roi_edge_coords)} edges")

        # Label connected components
        labeled_roi = measure.label(roi)
        regions = measure.regionprops(labeled_roi)

        # Filter regions based on anatomical location and size
        filtered_regions = [r for r in regions if 200 < r.area < 80000]

        half_sized_mask = np.zeros((height, width // 2), dtype=np.bool_)
        if filtered_regions:
            largest_region = max(filtered_regions, key=lambda r: r.area)
            if side == 'left':
                half_sized_mask[height // 3: 2 * height // 3, :width // 6] = (labeled_roi == largest_region.label)
            elif side == 'right':
                half_sized_mask[height // 3: 2 * height // 3, width // 6:] = (labeled_roi == largest_region.label)

        selected_edge_coords = np.argwhere(half_sized_mask)
        logging.info(f"Selected edge coordinates for {side} hippocampus: {len(selected_edge_coords)} edges")

        logging.info(f"{side.capitalize()} hippocampal region extracted with area: {np.sum(half_sized_mask)}")
        return half_sized_mask
    except Exception as e:
        logging.error(f"Error extracting hippocampal region: {e}")
        raise


def v80_extract_hippocampal_region(brain_image, edges, side='left'):
    try:
        logging.info(f"Extracting {side} hippocampal region")
        height, width = brain_image.shape

        logging.info(f"Total edges detected in the whole image: {np.argwhere(edges)}")

        # Define the ROI for the hippocampus focusing on the middle third both vertically and horizontally
        if side == 'left':
            roi = edges[height // 3: 2 * height // 3, width // 3: width // 2]
        elif side == 'right':
            roi = edges[height // 3: 2 * height // 3, width // 2: 2 * width // 3]
        else:
            raise ValueError("side must be 'left' or 'right'")

        roi_edge_coords = np.argwhere(roi)
        logging.info(f"Edges coordinates considered for {side} hippocampus: {roi_edge_coords}")

        # Label connected components
        labeled_roi = measure.label(roi)
        regions = measure.regionprops(labeled_roi)

        # Filter regions based on anatomical location and size
        filtered_regions = [r for r in regions if 200 < r.area < 80000]

        half_sized_mask = np.zeros((height, width // 2), dtype=np.bool_)
        if filtered_regions:
            largest_region = max(filtered_regions, key=lambda r: r.area)
            if side == 'left':
                half_sized_mask[height // 3: 2 * height // 3, width // 6: width // 6 + (width // 2 - width // 3)] = (labeled_roi == largest_region.label)
            elif side == 'right':
                half_sized_mask[height // 3: 2 * height // 3, width // 6: width // 6 + (width // 2 - width // 3)] = (labeled_roi == largest_region.label)

        selected_edge_coords = np.argwhere(half_sized_mask)
        logging.info(f"Selected edge coordinates for {side} hippocampus: {selected_edge_coords}")

        logging.info(f"{side.capitalize()} hippocampal region extracted with area: {np.sum(half_sized_mask)}")
        return half_sized_mask
    except Exception as e:
        logging.error(f"Error extracting hippocampal region: {e}")
        raise

def v79_extract_hippocampal_region(brain_image, edges, side='left'):
    try:
        logging.info(f"Extracting {side} hippocampal region")
        height, width = brain_image.shape

        logging.info(f"Total edges detected in the whole image: {np.argwhere(edges)}")

        # Define the ROI for the hippocampus focusing on the middle third both vertically and horizontally
        if side == 'left':
            roi = edges[height // 3: 2 * height // 3, width // 3: width // 2]
        elif side == 'right':
            roi = edges[height // 3: 2 * height // 3, width // 2: 2 * width // 3]
        else:
            raise ValueError("side must be 'left' or 'right'")

        roi_edge_coords = np.argwhere(roi)
        logging.info(f"Edges coordinates considered for {side} hippocampus: {roi_edge_coords}")

        # Label connected components
        labeled_roi = measure.label(roi)
        regions = measure.regionprops(labeled_roi)

        # Filter regions based on anatomical location and size
        filtered_regions = [r for r in regions if 200 < r.area < 80000]

        half_sized_mask = np.zeros((height, width // 2), dtype=np.bool_)
        if filtered_regions:
            largest_region = max(filtered_regions, key=lambda r: r.area)
            if side == 'left':
                half_sized_mask[height // 3: 2 * height // 3, width // 6: width // 3] = (labeled_roi == largest_region.label)
            elif side == 'right':
                half_sized_mask[height // 3: 2 * height // 3, width // 6: width // 3] = (labeled_roi == largest_region.label)

        selected_edge_coords = np.argwhere(half_sized_mask)
        logging.info(f"Selected edge coordinates for {side} hippocampus: {selected_edge_coords}")

        logging.info(f"{side.capitalize()} hippocampal region extracted with area: {np.sum(half_sized_mask)}")
        return half_sized_mask
    except Exception as e:
        logging.error(f"Error extracting hippocampal region: {e}")
        raise

def v77_extract_hippocampal_region(brain_image, edges, side='left'):
    try:
        logging.info(f"Extracting {side} hippocampal region")
        height, width = brain_image.shape

        logging.info(f"Total edges detected in the whole image: {np.argwhere(edges)}")

        # Define the ROI for the hippocampus focusing on the middle third both vertically and horizontally
        if side == 'left':
            roi = edges[height // 3: 2 * height // 3, width // 3: width // 2]
        elif side == 'right':
            roi = edges[height // 3: 2 * height // 3, width // 2: 2 * width // 3]
        else:
            raise ValueError("side must be 'left' or 'right'")

        roi_edge_coords = np.argwhere(roi)
        logging.info(f"Edges coordinates considered for {side} hippocampus: {roi_edge_coords}")

        # Label connected components
        labeled_roi = measure.label(roi)
        regions = measure.regionprops(labeled_roi)

        # Filter regions based on anatomical location and size
        filtered_regions = [r for r in regions if 100 < r.area < 2000]

        hippocampal_mask = np.zeros_like(edges)
        if filtered_regions:
            largest_region = max(filtered_regions, key=lambda r: r.area)
            hippocampal_mask[height // 3: 2 * height // 3, width // 3: width // 2] = (labeled_roi == largest_region.label)

        selected_edge_coords = np.argwhere(hippocampal_mask)
        logging.info(f"Selected edge coordinates for {side} hippocampus: {selected_edge_coords}")

        logging.info(f"{side.capitalize()} hippocampal region extracted with area: {np.sum(hippocampal_mask)}")
        return hippocampal_mask
    except Exception as e:
        logging.error(f"Error extracting hippocampal region: {e}")
        raise

def v75_extract_hippocampal_region(brain_image, edges, side='left'):
    try:
        logging.info(f"Extracting {side} hippocampal region")
        height, width = edges.shape

        logging.info(f"Total edges detected in the whole image: {np.argwhere(edges)}")

        # Define the ROI for the hippocampus focusing on the middle third both vertically and horizontally
        if side == 'left':
            roi = edges[height // 3: 2 * height // 3, width // 3: width // 2]
        elif side == 'right':
            roi = edges[height // 3: 2 * height // 3, width // 2: 2 * width // 3]
        else:
            raise ValueError("side must be 'left' or 'right'")

        roi_edge_coords = np.argwhere(roi)
        logging.info(f"Edges coordinates considered for {side} hippocampus: {roi_edge_coords}")

        # Label connected components
        labeled_roi = measure.label(roi)
        regions = measure.regionprops(labeled_roi)

        # Filter regions based on anatomical location and size
        filtered_regions = [r for r in regions if 100 < r.area < 2000]

        hippocampal_mask = np.zeros_like(roi)
        if filtered_regions:
            largest_region = max(filtered_regions, key=lambda r: r.area)
            hippocampal_mask = (labeled_roi == largest_region.label)

        selected_edge_coords = np.argwhere(hippocampal_mask)
        logging.info(f"Selected edge coordinates for {side} hippocampus: {selected_edge_coords}")

        logging.info(f"{side.capitalize()} hippocampal region extracted with area: {np.sum(hippocampal_mask)}")
        return hippocampal_mask
    except Exception as e:
        logging.error(f"Error extracting hippocampal region: {e}")
        raise


def v72_extract_hippocampal_region(brain_image, edges, side='left'):
    try:
        logging.info(f"Extracting {side} hippocampal region")
        height, width = edges.shape

        logging.info(f"Total edges detected in the whole image: {np.argwhere(edges)}")

        # Define the ROI for the hippocampus focusing on the middle third both vertically and horizontally
        if side == 'left':
            roi = edges[height // 3: 2 * height // 3, width // 3: width // 2]
        elif side == 'right':
            roi = edges[height // 3: 2 * height // 3, width // 2: 2 * width // 3]
        else:
            raise ValueError("side must be 'left' or 'right'")

        roi_edge_coords = np.argwhere(roi)
        logging.info(f"Edges coordinates considered for {side} hippocampus: {roi_edge_coords}")

        # Label connected components
        labeled_roi = measure.label(roi)
        regions = measure.regionprops(labeled_roi)

        # Filter regions based on anatomical location and size
        filtered_regions = [r for r in regions if 100 < r.area < 2000]

        hippocampal_mask = np.zeros_like(roi)
        if filtered_regions:
            largest_region = max(filtered_regions, key=lambda r: r.area)
            hippocampal_mask = (labeled_roi == largest_region.label)

        selected_edge_coords = np.argwhere(hippocampal_mask)
        logging.info(f"Selected edge coordinates for {side} hippocampus: {selected_edge_coords}")

        logging.info(f"{side.capitalize()} hippocampal region extracted with area: {np.sum(hippocampal_mask)}")
        return hippocampal_mask
    except Exception as e:
        logging.error(f"Error extracting hippocampal region: {e}")
        raise

def v70_extract_hippocampal_region(brain_image, edges, side='left'):
    try:
        logging.info(f"Extracting {side} hippocampal region")
        height, width = edges.shape

        logging.info(f"Total edges detected in the whole image: {np.argwhere(edges)}")

        # Define the ROI for the hippocampus focusing on the middle third both vertically and horizontally
        if side == 'left':
            roi = edges[height // 3: 2 * height // 3, width // 3: width // 2]
        elif side == 'right':
            roi = edges[height // 3: 2 * height // 3, width // 2: 2 * width // 3]
        else:
            raise ValueError("side must be 'left' or 'right'")

        roi_edge_coords = np.argwhere(roi)
        logging.info(f"Edges coordinates considered for {side} hippocampus: {roi_edge_coords}")

        # Label connected components
        labeled_roi = measure.label(roi)
        regions = measure.regionprops(labeled_roi)

        # Filter regions based on anatomical location and size
        filtered_regions = [r for r in regions if 100 < r.area < 2000]

        hippocampal_mask = np.zeros_like(roi)
        if filtered_regions:
            largest_region = max(filtered_regions, key=lambda r: r.area)
            hippocampal_mask = (labeled_roi == largest_region.label)

        selected_edge_coords = np.argwhere(hippocampal_mask)
        logging.info(f"Selected edge coordinates for {side} hippocampus: {selected_edge_coords}")

        logging.info(f"{side.capitalize()} hippocampal region extracted with area: {np.sum(hippocampal_mask)}")
        return hippocampal_mask
    except Exception as e:
        logging.error(f"Error extracting hippocampal region: {e}")
        raise

def v50_extract_hippocampal_region(brain_image, edges, side='left'):
    try:
        logging.info(f"Extracting {side} hippocampal region")
        height, width = edges.shape
        logging.info(f"dimensions: {height}, {width}")
        logging.info(f"Total edges detected in the whole image: {np.argwhere(edges)}")

        # Define the ROI for the hippocampus focusing on the middle third both vertically and horizontally
        if side == 'left':
            roi = edges[height // 3: 2 * height // 3, width // 3: width // 2]
        elif side == 'right':
            roi = edges[height // 3: 2 * height // 3, width // 2: 2 * width // 3]
        else:
            raise ValueError("side must be 'left' or 'right'")

        roi_edge_coords = np.argwhere(roi)
        logging.info(f"Edges coordinates considered for {side} hippocampus: {roi_edge_coords}")

        # Label connected components
        labeled_roi = measure.label(roi)
        regions = measure.regionprops(labeled_roi)

        # Filter regions based on anatomical location and size
        filtered_regions = [r for r in regions if 100 < r.area < 2000]

        hippocampal_mask = np.zeros_like(edges)
        if filtered_regions:
            largest_region = max(filtered_regions, key=lambda r: r.area)
            if side == 'left':
                hippocampal_mask[height // 3: 2 * height // 3, width // 3: width // 2] = (labeled_roi == largest_region.label)
            else:
                hippocampal_mask[height // 3: 2 * height // 3, width // 2: 2 * width // 3] = (labeled_roi == largest_region.label)

        selected_edge_coords = np.argwhere(hippocampal_mask)
        logging.info(f"Selected edge coordinates for {side} hippocampus: {selected_edge_coords}")

        logging.info(f"{side.capitalize()} hippocampal region extracted with area: {np.sum(hippocampal_mask)}")
        return hippocampal_mask
    except Exception as e:
        logging.error(f"Error extracting hippocampal region: {e}")
        raise

def v7_extract_hippocampal_region(brain_image, edges, side='left'):
    try:
        logging.info(f"Extracting {side} hippocampal region")
        height, width = edges.shape

        logging.info(f"Total edges detected in the whole image: {np.sum(edges)}")

        # Define the ROI for the hippocampus focusing on the middle third both vertically and horizontally
        # Define the ROI for the hippocampus focusing on the middle third both vertically and horizontally
        if side == 'left':
            roi = edges[height // 3: 2 * height // 3, width // 3: width // 2]
        elif side == 'right':
            roi = edges[height // 3: 2 * height // 3, width // 2: 2 * width // 3]
        else:
            raise ValueError("side must be 'left' or 'right'")

        roi_edge_coords = np.argwhere(roi)
        logging.info(f"Edges coordinates considered for {side} hippocampus: {roi_edge_coords}")

        # Label connected components
        labeled_roi = measure.label(roi)
        regions = measure.regionprops(labeled_roi)

        # Filter regions based on anatomical location and size
        filtered_regions = [r for r in regions if 100 < r.area < 2000]

        hippocampal_mask = np.zeros_like(edges)
        if filtered_regions:
            largest_region = max(filtered_regions, key=lambda r: r.area)
            if side == 'left':
                hippocampal_mask[height // 3: 2 * height // 3, width // 3: width // 2] = (labeled_roi == largest_region.label)
            else:
                hippocampal_mask[height // 3: 2 * height // 3, width // 2: 2 * width // 3] = (labeled_roi == largest_region.label)


        selected_edge_coords = np.argwhere(hippocampal_mask)
        logging.info(f"Selected edge coordinates for {side} hippocampus: {selected_edge_coords}")

        logging.info(f"{side.capitalize()} hippocampal region extracted with area: {np.sum(hippocampal_mask)}")
        return hippocampal_mask
    except Exception as e:
        logging.error(f"Error extracting hippocampal region: {e}")
        raise

def b6_extract_hippocampal_region(brain_image, edges, side='left'):
    try:
        logging.info(f"Extracting {side} hippocampal region")
        height, width = edges.shape

        logging.info(f"Total edges detected in the whole image: {np.sum(edges)}")

        # Define the ROI for the hippocampus focusing on the middle third both vertically and horizontally
        if side == 'left':
            roi = edges[height // 3: 2 * height // 3, width // 3: width // 2]
        elif side == 'right':
            roi = edges[height // 3: 2 * height // 3, width // 2: 2 * width // 3]


        roi_edge_coords = np.argwhere(roi)
        logging.info(f"Edges coordinates considered for {side} hippocampus: {roi_edge_coords}")


        # Label connected components
        labeled_roi = measure.label(roi)
        regions = measure.regionprops(labeled_roi)

        # Filter regions based on anatomical location and size
        filtered_regions = [r for r in regions if 300 < r.area < 4000]

        hippocampal_mask = np.zeros_like(edges)
        if filtered_regions:
            largest_region = max(filtered_regions, key=lambda r: r.area)
            if side == 'left':
                hippocampal_mask[height // 3: 2 * height // 3, width // 3: width // 2] = (labeled_roi == largest_region.label)
            else:
                hippocampal_mask[height // 3: 2 * height // 3, width // 2: 2 * width // 3] = (labeled_roi == largest_region.label)

        selected_edge_coords = np.argwhere(hippocampal_mask)
        logging.info(f"Selected edge coordinates for {side} hippocampus: {selected_edge_coords}")

        logging.info(f"{side.capitalize()} hippocampal region extracted with area: {np.sum(hippocampal_mask)}")
        return hippocampal_mask
    except Exception as e:
        logging.error(f"Error extracting hippocampal region: {e}")
        raise

def v4_extract_hippocampal_region(brain_image, edges, side='left'):
    try:
        logging.info(f"Extracting {side} hippocampal region")
        height, width = edges.shape

        # Define the ROI for the hippocampus focusing on the middle third both vertically and horizontally
        if side == 'left':
            roi = edges[height // 3: 2 * height // 3, width // 3: 2 * width // 3]
        elif side == 'right':
            roi = edges[height // 3: 2 * height // 3, width // 3: 2 * width // 3]
        else:
            raise ValueError("side must be 'left' or 'right'")

        # Label connected components
        labeled_roi = measure.label(roi)
        regions = measure.regionprops(labeled_roi)

        # Filter regions based on anatomical location and size
        filtered_regions = [r for r in regions if 100 < r.area < 2000]

        hippocampal_mask = np.zeros_like(edges)
        if filtered_regions:
            largest_region = max(filtered_regions, key=lambda r: r.area)
            if side == 'left':
                hippocampal_mask[height // 3: 2 * height // 3, width // 3: 2 * width // 3] = (labeled_roi == largest_region.label)
            else:
                hippocampal_mask[height // 3: 2 * height // 3, width // 3: 2 * width // 3] = (labeled_roi == largest_region.label)

        logging.info(f"{side.capitalize()} hippocampal region extracted")
        return hippocampal_mask
    except Exception as e:
        logging.error(f"Error extracting hippocampal region: {e}")
        raise

def v3_extract_hippocampal_region(brain_image, edges, side='left'):
    try:
        logging.info(f"Extracting {side} hippocampal region")
        height, width = edges.shape

        # Define the ROI for the hippocampus focusing on the middle third both vertically and horizontally
        if side == 'left':
            roi = edges[height // 3: 2 * height // 3, width // 6: width // 3]
        elif side == 'right':
            roi = edges[height // 3: 2 * height // 3, 2 * width // 3: 5 * width // 6]
        else:
            raise ValueError("side must be 'left' or 'right'")

        # Label connected components
        labeled_roi = measure.label(roi)
        regions = measure.regionprops(labeled_roi)

        # Filter regions based on anatomical location and size
        filtered_regions = [r for r in regions if 100 < r.area < 2000]

        hippocampal_mask = np.zeros_like(edges)
        if filtered_regions:
            largest_region = max(filtered_regions, key=lambda r: r.area)
            if side == 'left':
                hippocampal_mask[height // 3: 2 * height // 3, width // 6: width // 3] = (labeled_roi == largest_region.label)
            else:
                hippocampal_mask[height // 3: 2 * height // 3, 2 * width // 3: 5 * width // 6] = (labeled_roi == largest_region.label)

        logging.info(f"{side.capitalize()} hippocampal region extracted")
        return hippocampal_mask
    except Exception as e:
        logging.error(f"Error extracting hippocampal region: {e}")
        raise


def v2_extract_hippocampal_region(brain_image, edges, side='left'):
    try:
        logging.info(f"Extracting {side} hippocampal region")
        height, width = edges.shape

        # Define the ROI for the hippocampus
        if side == 'left':
            roi = edges[height // 3: 2 * height // 3, :width // 3]
        elif side == 'right':
            roi = edges[height // 3: 2 * height // 3, 2 * width // 3:]
        else:
            raise ValueError("side must be 'left' or 'right'")

        # Label connected components
        labeled_roi = measure.label(roi)
        regions = measure.regionprops(labeled_roi)

        # Filter regions based on anatomical location and size
        filtered_regions = [r for r in regions if 300 < r.area < 4000]

        hippocampal_mask = np.zeros_like(edges)
        if filtered_regions:
            largest_region = max(filtered_regions, key=lambda r: r.area)
            if side == 'left':
                hippocampal_mask[height // 3: 2 * height // 3, :width // 3] = (labeled_roi == largest_region.label)
            else:
                hippocampal_mask[height // 3: 2 * height // 3, 2 * width // 3:] = (labeled_roi == largest_region.label)

        logging.info(f"{side.capitalize()} hippocampal region extracted")
        return hippocampal_mask
    except Exception as e:
        logging.error(f"Error extracting hippocampal region: {e}")
        raise



def compare_hippocampal_regions(left_mask, right_mask):
    try:
        logging.info("Comparing hippocampal regions")
        left_area = np.sum(left_mask)
        right_area = np.sum(right_mask)
        logging.info(f"Left hippocampal region area: {left_area}")
        logging.info(f"Right hippocampal region area: {right_area}")
        return left_area, right_area
    except Exception as e:
        logging.error(f"Error during hippocampal region comparison: {e}")
        raise

def create_full_sized_mask(mask, height, width, side='left'):
    full_mask = np.zeros((height, width), dtype=np.bool_)
    try:
        mask_height, mask_width = mask.shape
        if side == 'left':
            full_mask[height // 3: 2 * height // 3, width // 3: width // 3 + mask_width] = mask
        elif side == 'right':
            full_mask[height // 3: 2 * height // 3, width // 2: width // 2 + mask_width] = mask
    except ValueError as e:
        logging.error(f"Error creating full {side} mask: {e}")
    return full_mask

def visualize_results(image, left_edges, right_edges, left_mask, right_mask):
    try:
        try:
            height, width = image.shape
        except Exception as e:
            print ("image is NoneType")
        logging.info(f"Image dimensions: {height}, {width}")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Original Image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        
        left_edges_colored = np.zeros((height, width // 2, 3), dtype=np.float32)
        right_edges_colored = np.zeros((height, width // 2, 3), dtype=np.float32)
        
        logging.info("Creating half-sized masks")
        
        #logging.info(f"Left mask shape: {left_mask.shape}, Right mask shape: {right_mask.shape}")

        logging.info("Assigning colors to edges and masks separately using np.where")
        # Assign colors to edges and masks separately using np.where
        try:
            left_edges_colored[..., 2] = np.where(left_edges, 1, 0)  # Blue for edges
        except Exception as e:
            logging.error(f"Error coloring left edges: {e}")
        
        try:
            left_edges_colored[..., 0] = np.where(left_mask & left_edges, 1, 0)  # Red for hippocampal edges
        except Exception as e:
            logging.error(f"Error coloring left hippocampal edges: {e}")

        try:
            right_edges_colored[..., 2] = np.where(right_edges, 1, 0)  # Blue for edges
        except Exception as e:
            logging.error(f"Error coloring right edges: {e}")
        
        try:
            right_edges_colored[..., 0] = np.where(right_mask & right_edges, 1, 0)  # Red for hippocampal edges
        except Exception as e:
            logging.error(f"Error coloring right hippocampal edges: {e}")
        
        logging.info(f"Left edges colored shape: {left_edges_colored.shape}, Right edges colored shape: {right_edges_colored.shape}")

        axes[0, 1].imshow(left_edges_colored)
        axes[0, 1].set_title('Left Hippocampus Edges')

        axes[1, 0].imshow(right_edges_colored)
        axes[1, 0].set_title('Right Hippocampus Edges')

        plt.show()
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        raise

def v89_visualize_results(image, left_edges, right_edges, left_mask, right_mask):
    try:
        height, width = image.shape
        logging.info(f"Image dimensions: {height}, {width}")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Original Image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        
        left_edges_colored = np.zeros((height, width // 2, 3), dtype=np.float32)
        right_edges_colored = np.zeros((height, width // 2, 3), dtype=np.float32)
        
        logging.info("Creating half-sized masks")
        
        logging.info(f"Left mask shape: {left_mask.shape}, Right mask shape: {right_mask.shape}")

        logging.info("Assigning colors to edges and masks separately using np.where")
        # Assign colors to edges and masks separately using np.where
        try:
            left_edges_colored[..., 2] = np.where(left_edges[:, :width // 2], 1, 0)  # Blue for edges
        except Exception as e:
            logging.error(f"Error coloring left edges: {e}")
        
        try:
            left_edges_colored[..., 0] = np.where(left_mask[:, :width // 2] & left_edges[:, :width // 2], 1, 0)  # Red for hippocampal edges
        except Exception as e:
            logging.error(f"Error coloring left hippocampal edges: {e}")

        try:
            right_edges_colored[..., 2] = np.where(right_edges[:, width // 2:], 1, 0)  # Blue for edges
        except Exception as e:
            logging.error(f"Error coloring right edges: {e}")
        
        try:
            right_edges_colored[..., 0] = np.where(right_mask[:, width // 2:] & right_edges[:, width // 2:], 1, 0)  # Red for hippocampal edges
        except Exception as e:
            logging.error(f"Error coloring right hippocampal edges: {e}")
        
        logging.info(f"Left edges colored shape: {left_edges_colored.shape}, Right edges colored shape: {right_edges_colored.shape}")

        axes[0, 1].imshow(left_edges_colored)
        axes[0, 1].set_title('Left Hippocampus Edges')

        axes[1, 0].imshow(right_edges_colored)
        axes[1, 0].set_title('Right Hippocampus Edges')

        plt.show()
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        raise

def v75_visualize_results(image, left_edges, right_edges, left_mask, right_mask):
    try:
        height, width = image.shape
        logging.info(f"Image dimensions: {height}, {width}")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original Image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        
        left_edges_colored = np.zeros((height, width, 3), dtype=np.float32)
        right_edges_colored = np.zeros((height, width, 3), dtype=np.float32)
        
        logging.info("Creating full-sized masks")
        # Create full-sized masks
        full_left_mask = create_full_sized_mask(left_mask, height, width, side='left')
        full_right_mask = create_full_sized_mask(right_mask, height, width, side='right')
        
        logging.info(f"Left mask shape: {full_left_mask.shape}, Right mask shape: {full_right_mask.shape}")

        logging.info("Assigning colors to edges and masks separately using np.where")
        # Assign colors to edges and masks separately using np.where
        try:
            left_edges_colored[..., 2] = np.where(left_edges, 1, 0)  # Blue for edges
        except Exception as e:
            logging.error(f"Error coloring left edges: {e}")
        
        try:
            left_edges_colored[..., 0] = np.where(full_left_mask & left_edges, 1, 0)  # Red for hippocampal edges
        except Exception as e:
            logging.error(f"Error coloring left hippocampal edges: {e}")

        try:
            right_edges_colored[..., 2] = np.where(right_edges, 1, 0)  # Blue for edges
        except Exception as e:
            logging.error(f"Error coloring right edges: {e}")
        
        try:
            right_edges_colored[..., 0] = np.where(full_right_mask & right_edges, 1, 0)  # Red for hippocampal edges
        except Exception as e:
            logging.error(f"Error coloring right hippocampal edges: {e}")
        
        logging.info(f"Left edges colored shape: {left_edges_colored.shape}, Right edges colored shape: {right_edges_colored.shape}")

        axes[0, 1].imshow(left_edges_colored)
        axes[0, 1].set_title('Left Hippocampus Edges')

        axes[0, 2].imshow(right_edges_colored)
        axes[0, 2].set_title('Right Hippocampus Edges')

        logging.info("Adjusting the masks to match the original image dimensions")
        # Adjust the masks to match the original image dimensions
        try:
            combined_overlay = np.copy(image)
            combined_overlay = np.stack([combined_overlay]*3, axis=-1)  # Convert to RGB
        except Exception as e:
            logging.error(f"Error adjusting masks: {e}")

        logging.info(f"Combined overlay shape: {combined_overlay.shape}")

        logging.info("Applying hippocampal edges in red and other edges in blue")
        try:
            combined_overlay[full_left_mask & left_edges, 0] = 1  # Red for left hippocampal edges
        except Exception as e:
            logging.error(f"Error applying left hippocampal edges: {e}")
        
        try:
            combined_overlay[full_right_mask & right_edges, 0] = 1  # Red for right hippocampal edges
        except Exception as e:
            logging.error(f"Error applying right hippocampal edges: {e}")

        try:
            combined_overlay[~full_left_mask & left_edges, 2] = 1  # Blue for other left edges
        except Exception as e:
            logging.error(f"Error applying other left edges: {e}")
        
        try:
            combined_overlay[~full_right_mask & right_edges, 2] = 1  # Blue for other right edges
        except Exception as e:
            logging.error(f"Error applying other right edges: {e}")

        logging.info(f"Final combined overlay shape: {combined_overlay.shape}")

        axes[1, 1].imshow(combined_overlay)
        axes[1, 1].set_title('Hippocampal Regions Overlay')

        plt.show()
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        raise

def v73_visualize_results(image, left_edges, right_edges, left_mask, right_mask):
    try:
        height, width = image.shape
        logging.info(f"Image dimensions: {height}, {width}")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original Image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        
        left_edges_colored = np.zeros((height, width, 3), dtype=np.float32)
        right_edges_colored = np.zeros((height, width, 3), dtype=np.float32)
        
        logging.info("Creating full-sized masks")
        # Create full-sized masks
        full_left_mask = create_full_sized_mask(left_mask, height, width, side='left')
        full_right_mask = create_full_sized_mask(right_mask, height, width, side='right')
        
        logging.info(f"Left mask shape: {full_left_mask.shape}, Right mask shape: {full_right_mask.shape}")

        logging.info("Assigning colors to edges and masks separately using np.where")
        # Assign colors to edges and masks separately using np.where
        try:
            left_edges_colored[..., 2] = np.where(left_edges, 1, 0)  # Blue for edges
        except Exception as e:
            logging.error(f"Error coloring left edges: {e}")
        
        try:
            left_edges_colored[..., 0] = np.where(full_left_mask & left_edges, 1, 0)  # Red for hippocampal edges
        except Exception as e:
            logging.error(f"Error coloring left hippocampal edges: {e}")

        try:
            right_edges_colored[..., 2] = np.where(right_edges, 1, 0)  # Blue for edges
        except Exception as e:
            logging.error(f"Error coloring right edges: {e}")
        
        try:
            right_edges_colored[..., 0] = np.where(full_right_mask & right_edges, 1, 0)  # Red for hippocampal edges
        except Exception as e:
            logging.error(f"Error coloring right hippocampal edges: {e}")
        
        logging.info(f"Left edges colored shape: {left_edges_colored.shape}, Right edges colored shape: {right_edges_colored.shape}")

        axes[0, 1].imshow(left_edges_colored)
        axes[0, 1].set_title('Left Hippocampus Edges')

        axes[0, 2].imshow(right_edges_colored)
        axes[0, 2].set_title('Right Hippocampus Edges')

        logging.info("Adjusting the masks to match the original image dimensions")
        # Adjust the masks to match the original image dimensions
        try:
            combined_overlay = np.copy(image)
            combined_overlay = np.stack([combined_overlay]*3, axis=-1)  # Convert to RGB
        except Exception as e:
            logging.error(f"Error adjusting masks: {e}")

        logging.info(f"Combined overlay shape: {combined_overlay.shape}")

        logging.info("Applying hippocampal edges in red and other edges in blue")
        try:
            combined_overlay[full_left_mask & left_edges, 0] = 1  # Red for left hippocampal edges
        except Exception as e:
            logging.error(f"Error applying left hippocampal edges: {e}")
        
        try:
            combined_overlay[full_right_mask & right_edges, 0] = 1  # Red for right hippocampal edges
        except Exception as e:
            logging.error(f"Error applying right hippocampal edges: {e}")

        try:
            combined_overlay[~full_left_mask & left_edges, 2] = 1  # Blue for other left edges
        except Exception as e:
            logging.error(f"Error applying other left edges: {e}")
        
        try:
            combined_overlay[~full_right_mask & right_edges, 2] = 1  # Blue for other right edges
        except Exception as e:
            logging.error(f"Error applying other right edges: {e}")

        logging.info(f"Final combined overlay shape: {combined_overlay.shape}")

        axes[1, 1].imshow(combined_overlay)
        axes[1, 1].set_title('Hippocampal Regions Overlay')

        plt.show()
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        raise

def v71_visualize_results(image, left_edges, right_edges, left_mask, right_mask):
    try:
        height, width = image.shape
        logging.info(f"Image dimensions: {height}, {width}")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original Image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        
        left_edges_colored = np.zeros((height, width, 3), dtype=np.float32)
        right_edges_colored = np.zeros((height, width, 3), dtype=np.float32)
        
        logging.info("Creating full-sized masks")
        # Create full-sized masks
        full_left_mask = create_full_sized_mask(left_mask, height, width, side='left')
        full_right_mask = create_full_sized_mask(right_mask, height, width, side='right')
        
        logging.info(f"Left mask shape: {full_left_mask.shape}, Right mask shape: {full_right_mask.shape}")

        logging.info("Assigning colors to edges and masks separately using np.where")
        # Assign colors to edges and masks separately using np.where
        try:
            left_edges_colored[..., 2] = np.where(left_edges, 1, 0)  # Blue for edges
        except Exception as e:
            logging.error(f"Error coloring left edges: {e}")
        
        try:
            left_edges_colored[..., 0] = np.where(full_left_mask & left_edges, 1, 0)  # Red for hippocampal edges
        except Exception as e:
            logging.error(f"Error coloring left hippocampal edges: {e}")

        try:
            right_edges_colored[..., 2] = np.where(right_edges, 1, 0)  # Blue for edges
        except Exception as e:
            logging.error(f"Error coloring right edges: {e}")
        
        try:
            right_edges_colored[..., 0] = np.where(full_right_mask & right_edges, 1, 0)  # Red for hippocampal edges
        except Exception as e:
            logging.error(f"Error coloring right hippocampal edges: {e}")
        
        logging.info(f"Left edges colored shape: {left_edges_colored.shape}, Right edges colored shape: {right_edges_colored.shape}")

        axes[0, 1].imshow(left_edges_colored)
        axes[0, 1].set_title('Left Hippocampus Edges')

        axes[0, 2].imshow(right_edges_colored)
        axes[0, 2].set_title('Right Hippocampus Edges')

        logging.info("Adjusting the masks to match the original image dimensions")
        # Adjust the masks to match the original image dimensions
        try:
            combined_overlay = np.copy(image)
            combined_overlay = np.stack([combined_overlay]*3, axis=-1)  # Convert to RGB
        except Exception as e:
            logging.error(f"Error adjusting masks: {e}")

        logging.info(f"Combined overlay shape: {combined_overlay.shape}")

        logging.info("Applying hippocampal edges in red and other edges in blue")
        try:
            combined_overlay[full_left_mask & left_edges, 0] = 1  # Red for left hippocampal edges
        except Exception as e:
            logging.error(f"Error applying left hippocampal edges: {e}")
        
        try:
            combined_overlay[full_right_mask & right_edges, 0] = 1  # Red for right hippocampal edges
        except Exception as e:
            logging.error(f"Error applying right hippocampal edges: {e}")

        try:
            combined_overlay[~full_left_mask & left_edges, 2] = 1  # Blue for other left edges
        except Exception as e:
            logging.error(f"Error applying other left edges: {e}")
        
        try:
            combined_overlay[~full_right_mask & right_edges, 2] = 1  # Blue for other right edges
        except Exception as e:
            logging.error(f"Error applying other right edges: {e}")

        logging.info(f"Final combined overlay shape: {combined_overlay.shape}")

        axes[1, 1].imshow(combined_overlay)
        axes[1, 1].set_title('Hippocampal Regions Overlay')

        plt.show()
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        raise

def v60_visualize_results(image, left_edges, right_edges, left_mask, right_mask):
    try:
        height, width = image.shape
        logging.info(f"Image dimensions: {height}, {width}")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original Image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        
        left_edges_colored = np.zeros((height, width, 3), dtype=np.float32)
        right_edges_colored = np.zeros((height, width, 3), dtype=np.float32)
        
        logging.info("Creating full-sized masks")
        # Create full-sized masks
        full_left_mask = create_full_sized_mask(left_mask, height, width, side='left')
        full_right_mask = create_full_sized_mask(right_mask, height, width, side='right')
        
        logging.info(f"Left mask shape: {full_left_mask.shape}, Right mask shape: {full_right_mask.shape}")

        logging.info("Assigning colors to edges and masks separately using np.where")
        # Assign colors to edges and masks separately using np.where
        try:
            left_edges_colored[..., 2] = np.where(left_edges, 1, 0)  # Blue for edges
        except Exception as e:
            logging.error(f"Error coloring left edges: {e}")
        
        try:
            left_edges_colored[..., 0] = np.where(full_left_mask & left_edges, 1, 0)  # Red for hippocampal edges
        except Exception as e:
            logging.error(f"Error coloring left hippocampal edges: {e}")

        try:
            right_edges_colored[..., 2] = np.where(right_edges, 1, 0)  # Blue for edges
        except Exception as e:
            logging.error(f"Error coloring right edges: {e}")
        
        try:
            right_edges_colored[..., 0] = np.where(full_right_mask & right_edges, 1, 0)  # Red for hippocampal edges
        except Exception as e:
            logging.error(f"Error coloring right hippocampal edges: {e}")
        
        logging.info(f"Left edges colored shape: {left_edges_colored.shape}, Right edges colored shape: {right_edges_colored.shape}")

        axes[0, 1].imshow(left_edges_colored)
        axes[0, 1].set_title('Left Hippocampus Edges')

        axes[0, 2].imshow(right_edges_colored)
        axes[0, 2].set_title('Right Hippocampus Edges')

        logging.info("Adjusting the masks to match the original image dimensions")
        # Adjust the masks to match the original image dimensions
        try:
            combined_overlay = np.copy(image)
            combined_overlay = np.stack([combined_overlay]*3, axis=-1)  # Convert to RGB
        except Exception as e:
            logging.error(f"Error adjusting masks: {e}")

        logging.info(f"Combined overlay shape: {combined_overlay.shape}")

        logging.info("Applying hippocampal edges in red and other edges in blue")
        try:
            combined_overlay[full_left_mask & left_edges, 0] = 1  # Red for left hippocampal edges
        except Exception as e:
            logging.error(f"Error applying left hippocampal edges: {e}")
        
        try:
            combined_overlay[full_right_mask & right_edges, 0] = 1  # Red for right hippocampal edges
        except Exception as e:
            logging.error(f"Error applying right hippocampal edges: {e}")

        try:
            combined_overlay[~full_left_mask & left_edges, 2] = 1  # Blue for other left edges
        except Exception as e:
            logging.error(f"Error applying other left edges: {e}")
        
        try:
            combined_overlay[~full_right_mask & right_edges, 2] = 1  # Blue for other right edges
        except Exception as e:
            logging.error(f"Error applying other right edges: {e}")

        logging.info(f"Final combined overlay shape: {combined_overlay.shape}")

        axes[1, 1].imshow(combined_overlay)
        axes[1, 1].set_title('Hippocampal Regions Overlay')

        plt.show()
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        raise


def v50_visualize_results(image, left_edges, right_edges, left_mask, right_mask):
    try:
        height, width = image.shape
        logging.info(f"Image dimensions: {height}, {width}")
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        
        left_edges_colored = np.zeros((height, width, 3), dtype=np.float32)
        right_edges_colored = np.zeros((height, width, 3), dtype=np.float32)
        
        logging.info("Creating full-sized masks")
        # Create full-sized masks
        full_left_mask = np.zeros((height, width), dtype=np.bool_)
        full_right_mask = np.zeros((height, width), dtype=np.bool_)
        
        logging.info("Placing the half-sized masks in their respective locations")
        # Place the half-sized masks in their respective locations
        try:
            full_left_mask[height // 3: 2 * height // 3, width // 3: width // 2] = left_mask
        except Exception as e:
            logging.error(f"Error placing left mask: {e}")
        
        try:
            full_right_mask[height // 3: 2 * height // 3, width // 2: 2 * width // 3] = right_mask
        except Exception as e:
            logging.error(f"Error placing right mask: {e}")
        
        logging.info(f"Left mask shape: {full_left_mask.shape}, Right mask shape: {full_right_mask.shape}")

        logging.info("Assigning colors to edges and masks separately using np.where")
        # Assign colors to edges and masks separately using np.where
        try:
            left_edges_colored[..., 2] = np.where(left_edges, 1, 0)  # Blue for edges
        except Exception as e:
            logging.error(f"Error coloring left edges: {e}")
        
        try:
            left_edges_colored[..., 0] = np.where(full_left_mask & left_edges, 1, 0)  # Red for hippocampal edges
        except Exception as e:
            logging.error(f"Error coloring left hippocampal edges: {e}")

        try:
            right_edges_colored[..., 2] = np.where(right_edges, 1, 0)  # Blue for edges
        except Exception as e:
            logging.error(f"Error coloring right edges: {e}")
        
        try:
            right_edges_colored[..., 0] = np.where(full_right_mask & right_edges, 1, 0)  # Red for hippocampal edges
        except Exception as e:
            logging.error(f"Error coloring right hippocampal edges: {e}")
        
        logging.info(f"Left edges colored shape: {left_edges_colored.shape}, Right edges colored shape: {right_edges_colored.shape}")

        axes[0, 1].imshow(left_edges_colored)
        axes[0, 1].set_title('Left Hippocampus Edges')

        axes[1, 0].imshow(right_edges_colored)
        axes[1, 0].set_title('Right Hippocampus Edges')

        logging.info("Adjusting the masks to match the original image dimensions")
        # Adjust the masks to match the original image dimensions
        combined_overlay = np.copy(image)
        combined_overlay = np.stack([combined_overlay]*3, axis=-1)  # Convert to RGB

        logging.info(f"Combined overlay shape: {combined_overlay.shape}")

        logging.info("Applying hippocampal edges in red and other edges in blue")
        try:
            combined_overlay[full_left_mask & left_edges, 0] = 1  # Red for left hippocampal edges
        except Exception as e:
            logging.error(f"Error applying left hippocampal edges: {e}")
        
        try:
            combined_overlay[full_right_mask & right_edges, 0] = 1  # Red for right hippocampal edges
        except Exception as e:
            logging.error(f"Error applying right hippocampal edges: {e}")

        try:
            combined_overlay[~full_left_mask & left_edges, 2] = 1  # Blue for other left edges
        except Exception as e:
            logging.error(f"Error applying other left edges: {e}")
        
        try:
            combined_overlay[~full_right_mask & right_edges, 2] = 1  # Blue for other right edges
        except Exception as e:
            logging.error(f"Error applying other right edges: {e}")

        logging.info(f"Final combined overlay shape: {combined_overlay.shape}")

        axes[1, 1].imshow(combined_overlay)
        axes[1, 1].set_title('Hippocampal Regions Overlay')

        plt.show()
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        raise

def v30_visualize_results(image, left_edges, right_edges, left_mask, right_mask):
    try:
        height, width = image.shape
        logging.info(f"Image dimensions: {height}, {width}")

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')

        left_edges_colored = np.zeros((height, width, 3), dtype=np.float32)
        right_edges_colored = np.zeros((height, width, 3), dtype=np.float32)

        logging.info("Creating full-sized masks")
        # Create full-sized masks
        full_left_mask = np.zeros((height, width), dtype=np.bool_)
        full_right_mask = np.zeros((height, width), dtype=np.bool_)

        logging.info("Placing the half-sized masks in their respective locations")
        # Place the half-sized masks in their respective locations
        full_left_mask[height // 3: 2 * height // 3, width // 3: width // 2] = left_mask[height // 3: 2 * height // 3, width // 3: width // 2]
        full_right_mask[height // 3: 2 * height // 3, width // 2: 2 * width // 3] = right_mask[height // 3: 2 * height // 3, width // 2: 2 * width // 3]

        logging.info(f"Left mask shape: {full_left_mask.shape}, Right mask shape: {full_right_mask.shape}")

        logging.info("Assigning colors to edges and masks separately using np.where")
        # Assign colors to edges and masks separately using np.where
        try:
            left_edges_colored[..., 2] = np.where(left_edges, 1, 0)  # Blue for edges
        except Exception as e:
            logging.error(f"Error during step: {e}")
        logging.info("Did color left")
        try:
            left_edges_colored[..., 0] = np.where(left_mask & left_edges, 1, 0)  # Red for hippocampal edges
        except Exception as e:
            logging.error(f"Error during 2: {e}")
        logging.info("Did Red for hippocampal") 
        try:
            right_edges_colored[..., 2] = np.where(right_edges, 1, 0)  # Blue for edges
            right_edges_colored[..., 0] = np.where(right_mask & right_edges, 1, 0)  # Red for hippocampal edges
        except Exception as e:
            logging.error(f"Happened again: {e}")
        logging.info(f"Left edges colored shape: {left_edges_colored.shape}, Right edges colored shape: {right_edges_colored.shape}")

        axes[0, 1].imshow(left_edges_colored)
        axes[0, 1].set_title('Left Hippocampus Edges')

        axes[1, 0].imshow(right_edges_colored)
        axes[1, 0].set_title('Right Hippocampus Edges')

        logging.info("Adjusting the masks to match the original image dimensions")
        # Adjust the masks to match the original image dimensions
        try:
            combined_overlay = np.copy(image)
        except Exception as e:
            logging.error(f"np.copy(image): {e}")
        try:
            combined_overlay = np.stack([combined_overlay]*3, axis=-1)  # Convert to RGB
        except Exception as e:
            logging.error(f"np.stack([combined_overlay]*3: {e}")
              
        logging.info(f"Combined overlay shape: {combined_overlay.shape}")

        logging.info("Applying hippocampal edges in red and other edges in blue")
        # Apply hippocampal edges in red and other edges in blue
        try: 
            combined_overlay[full_left_mask & left_edges, 0] = 1  # Red for left hippocampal edges
            combined_overlay[full_right_mask & right_edges, 0] = 1  # Red for right hippocampal edges

            combined_overlay[~full_left_mask & left_edges, 2] = 1  # Blue for other left edges
            combined_overlay[~full_right_mask & right_edges, 2] = 1  # Blue for other right edges
        except Exception as e:
            logging.error(f"combined_overlay array manipulation: {e}")
        logging.info(f"Final combined overlay shape: {combined_overlay.shape}")

        axes[1, 1].imshow(combined_overlay)
        axes[1, 1].set_title('Hippocampal Regions Overlay')

        plt.show()
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        raise



def v21_visualize_results(image, left_edges, right_edges, left_mask, right_mask):
    try:
        height, width = image.shape
        logging.info(f"Image dimensions: {height}, {width}")

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')

        left_edges_colored = np.zeros((height, width, 3), dtype=np.float32)
        right_edges_colored = np.zeros((height, width, 3), dtype=np.float32)

        # Create full-sized masks
        full_left_mask = np.zeros((height, width), dtype=np.bool_)
        full_right_mask = np.zeros((height, width), dtype=np.bool_)

        # Place the half-sized masks in their respective locations
        full_left_mask[height // 3: 2 * height // 3, width // 3: width // 2] = left_mask[height // 3: 2 * height // 3, width // 3: width // 2]
        full_right_mask[height // 3: 2 * height // 3, width // 2: 2 * width // 3] = right_mask[height // 3: 2 * height // 3, width // 2: 2 * width // 3]

        # Assign colors to edges and masks separately using np.where
        left_edges_colored[..., 2] = np.where(left_edges, 1, 0)  # Blue for edges
        left_edges_colored[..., 0] = np.where(full_left_mask & left_edges, 1, 0)  # Red for hippocampal edges

        right_edges_colored[..., 2] = np.where(right_edges, 1, 0)  # Blue for edges
        right_edges_colored[..., 0] = np.where(full_right_mask & right_edges, 1, 0)  # Red for hippocampal edges

        axes[0, 1].imshow(left_edges_colored)
        axes[0, 1].set_title('Left Hippocampus Edges')

        axes[1, 0].imshow(right_edges_colored)
        axes[1, 0].set_title('Right Hippocampus Edges')

        # Adjust the masks to match the original image dimensions
        combined_overlay = np.copy(image)
        combined_overlay = np.stack([combined_overlay]*3, axis=-1)  # Convert to RGB

        logging.info(f"Left mask shape: {full_left_mask.shape}, Right mask shape: {full_right_mask.shape}")

        # Apply hippocampal edges in red and other edges in blue
        combined_overlay[full_left_mask & left_edges, 0] = 1  # Red for left hippocampal edges
        combined_overlay[full_right_mask & right_edges, 0] = 1  # Red for right hippocampal edges

        combined_overlay[~full_left_mask & left_edges, 2] = 1  # Blue for other left edges
        combined_overlay[~full_right_mask & right_edges, 2] = 1  # Blue for other right edges

        axes[1, 1].imshow(combined_overlay)
        axes[1, 1].set_title('Hippocampal Regions Overlay')

        plt.show()
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        raise

def v14_visualize_results(image, left_edges, right_edges, left_mask, right_mask):
    try:
        height, width = image.shape
        logging.info(f"Image dimensions: {height}, {width}")

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')

        left_edges_colored = np.zeros((height, width, 3), dtype=np.float32)
        right_edges_colored = np.zeros((height, width, 3), dtype=np.float32)

        # Assign colors to edges and masks separately using np.where
        left_edges_colored[..., 2] = np.where(left_edges, 1, 0)  # Blue for edges
        left_edges_colored[..., 0] = np.where(left_mask & left_edges, 1, 0)  # Red for hippocampal edges

        right_edges_colored[..., 2] = np.where(right_edges, 1, 0)  # Blue for edges
        right_edges_colored[..., 0] = np.where(right_mask & right_edges, 1, 0)  # Red for hippocampal edges

        axes[0, 1].imshow(left_edges_colored)
        axes[0, 1].set_title('Left Hippocampus Edges')

        axes[1, 0].imshow(right_edges_colored)
        axes[1, 0].set_title('Right Hippocampus Edges')

        # Adjust the masks to match the original image dimensions
        combined_overlay = np.copy(image)
        combined_overlay = np.stack([combined_overlay]*3, axis=-1)  # Convert to RGB

        logging.info(f"Left mask shape: {left_mask.shape}, Right mask shape: {right_mask.shape}")

        # Apply hippocampal edges in red and other edges in blue
        combined_overlay[height // 3: 2 * height // 3, width // 3: width // 2, 0] = np.where(left_mask[height // 3: 2 * height // 3, width // 3: width // 2] & left_edges[height // 3: 2 * height // 3, width // 3: width // 2], 1, combined_overlay[height // 3: 2 * height // 3, width // 3: width // 2, 0])
        combined_overlay[height // 3: 2 * height // 3, width // 2: 2 * width // 3, 0] = np.where(right_mask[height // 3: 2 * height // 3, width // 2: 2 * width // 3] & right_edges[height // 3: 2 * height // 3, width // 2: 2 * width // 3], 1, combined_overlay[height // 3: 2 * height // 3, width // 2: 2 * width // 3, 0])

        combined_overlay[height // 3: 2 * height // 3, width // 3: width // 2, 2] = np.where(~left_mask[height // 3: 2 * height // 3, width // 3: width // 2] & left_edges[height // 3: 2 * height // 3, width // 3: width // 2], 1, combined_overlay[height // 3: 2 * height // 3, width // 3: width // 2, 2])
        combined_overlay[height // 3: 2 * height // 3, width // 2: 2 * width // 3, 2] = np.where(~right_mask[height // 3: 2 * height // 3, width // 2: 2 * width // 3] & right_edges[height // 3: 2 * height // 3, width // 2: 2 * width // 3], 1, combined_overlay[height // 3: 2 * height // 3, width // 2: 2 * width // 3, 2])

        axes[1, 1].imshow(combined_overlay)
        axes[1, 1].set_title('Hippocampal Regions Overlay')

        plt.show()
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        raise


def v12_visualize_results(image, left_edges, right_edges, left_mask, right_mask):
    try:
        height, width = image.shape
        width_canvas = int(width/2)
        logging.info(f"Image dimensions: {height}, {width}")

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')

        left_edges_colored = np.zeros((height, width, 3), dtype=np.float32)
        right_edges_colored = np.zeros((height, width, 3), dtype=np.float32)

        # Assign colors to edges and masks separately using np.where
        left_edges_colored[..., 2] = np.where(left_edges, 1, 0)  # Blue for edges
        left_edges_colored[..., 0] = np.where(left_mask & left_edges, 1, 0)  # Red for hippocampal edges

        right_edges_colored[..., 2] = np.where(right_edges, 1, 0)  # Blue for edges
        right_edges_colored[..., 0] = np.where(right_mask & right_edges, 1, 0)  # Red for hippocampal edges

        axes[0, 1].imshow(left_edges_colored)
        axes[0, 1].set_title('Left Hippocampus Edges')

        axes[1, 0].imshow(right_edges_colored)
        axes[1, 0].set_title('Right Hippocampus Edges')

        # Adjust the masks to match the original image dimensions
        combined_overlay = np.copy(image)
        combined_overlay = np.stack([combined_overlay]*3, axis=-1)  # Convert to RGB

        logging.info(f"Left mask shape: {left_mask.shape}, Right mask shape: {right_mask.shape}")

        # Apply hippocampal edges in red and other edges in blue
        left_overlay = np.copy(combined_overlay)
        right_overlay = np.copy(combined_overlay)

        left_overlay[height // 3: 2 * height // 3, width // 3: width // 2, 0] = np.where(left_mask[height // 3: 2 * height // 3, width // 3: width // 2] & left_edges[height // 3: 2 * height // 3, width // 3: width // 2], 1, left_overlay[height // 3: 2 * height // 3, width // 3: width // 2, 0])
        right_overlay[height // 3: 2 * height // 3, width // 2: 2 * width // 3, 0] = np.where(right_mask[height // 3: 2 * height // 3, width // 2: 2 * width // 3] & right_edges[height // 3: 2 * height // 3, width // 2: 2 * width // 3], 1, right_overlay[height // 3: 2 * height // 3, width // 2: 2 * width // 3, 0])

        left_overlay[height // 3: 2 * height // 3, width // 3: width // 2, 2] = np.where(~left_mask[height // 3: 2 * height // 3, width // 3: width // 2] & left_edges[height // 3: 2 * height // 3, width // 3: width // 2], 1, left_overlay[height // 3: 2 * height // 3, width // 3: width // 2, 2])
        right_overlay[height // 3: 2 * height // 3, width // 2: 2 * width // 3, 2] = np.where(~right_mask[height // 3: 2 * height // 3, width // 2: 2 * width // 3] & right_edges[height // 3: 2 * height // 3, width // 2: 2 * width // 3], 1, right_overlay[height // 3: 2 * height // 3, width // 2: 2 * width // 3, 2])

        combined_overlay[height // 3: 2 * height // 3, width // 3: width // 2] = left_overlay[height // 3: 2 * height // 3, width // 3: width // 2]
        combined_overlay[height // 3: 2 * height // 3, width // 2: 2 * width // 3] = right_overlay[height // 3: 2 * height // 3, width // 2: 2 * width // 3]

        axes[1, 1].imshow(combined_overlay)
        axes[1, 1].set_title('Hippocampal Regions Overlay')

        plt.show()
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        raise

def v11_visualize_results(image, left_edges, right_edges, left_mask, right_mask):
    try:
        height, width = image.shape

        logging.info(f"dimensions: {height}, {width}")
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')

        left_edges_colored = np.zeros((height, width, 3), dtype=np.float32)
        right_edges_colored = np.zeros((height, width, 3), dtype=np.float32)

        # Assign colors to edges and masks separately using np.where
        left_edges_colored[..., 2] = np.where(left_edges, 1, 0)  # Blue for edges
        left_edges_colored[..., 0] = np.where(left_mask & left_edges, 1, 0)  # Red for hippocampal edges

        right_edges_colored[..., 2] = np.where(right_edges, 1, 0)  # Blue for edges
        right_edges_colored[..., 0] = np.where(right_mask & right_edges, 1, 0)  # Red for hippocampal edges

        axes[0, 1].imshow(left_edges_colored)
        axes[0, 1].set_title('Left Hippocampus Edges')

        axes[1, 0].imshow(right_edges_colored)
        axes[1, 0].set_title('Right Hippocampus Edges')

        # Adjust the masks to match the original image dimensions
        combined_overlay = np.copy(image)
        combined_overlay = np.stack([combined_overlay]*3, axis=-1)  # Convert to RGB

        logging.info(f"Left mask shape: {left_mask.shape}, Right mask shape: {right_mask.shape}")

        # Apply hippocampal edges in red and other edges in blue
        combined_overlay[height // 3: 2 * height // 3, :width // 3, 0] = np.where(left_mask[height // 3: 2 * height // 3, :width // 3] & left_edges[height // 3: 2 * height // 3, :width // 3], 1, combined_overlay[height // 3: 2 * height // 3, :width // 3, 0])
        combined_overlay[height // 3: 2 * height // 3, 2 * width // 3:, 0] = np.where(right_mask[height // 3: 2 * height // 3, 2 * width // 3:] & right_edges[height // 3: 2 * height // 3, 2 * width // 3:], 1, combined_overlay[height // 3: 2 * height // 3, 2 * width // 3:, 0])

        combined_overlay[height // 3: 2 * height // 3, :width // 3, 2] = np.where(~left_mask[height // 3: 2 * height // 3, :width // 3] & left_edges[height // 3: 2 * height // 3, :width // 3], 1, combined_overlay[height // 3: 2 * height // 3, :width // 3, 2])
        combined_overlay[height // 3: 2 * height // 3, 2 * width // 3:, 2] = np.where(~right_mask[height // 3: 2 * height // 3, 2 * width // 3:] & right_edges[height // 3: 2 * height // 3, 2 * width // 3:], 1, combined_overlay[height // 3: 2 * height // 3, 2 * width // 3:, 2])

        axes[1, 1].imshow(combined_overlay)
        axes[1, 1].set_title('Hippocampal Regions Overlay')

        plt.show()
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        raise

def v10_visualize_results(image, left_edges, right_edges, left_mask, right_mask):
    try:
        height, width = image.shape

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')

        left_edges_colored = np.zeros((height, width // 2, 3), dtype=np.float32)
        right_edges_colored = np.zeros((height, width // 2, 3), dtype=np.float32)

        # Assign colors to edges and masks separately using np.where
        left_edges_colored[..., 2] = np.where(left_edges[:, :width // 2], 1, 0)  # Blue for edges
        left_edges_colored[..., 0] = np.where(left_mask[:, :width // 2] & left_edges[:, :width // 2], 1, left_edges_colored[..., 0])  # Red for hippocampal edges

        right_edges_colored[..., 2] = np.where(right_edges[:, width // 2:], 1, 0)  # Blue for edges
        right_edges_colored[..., 0] = np.where(right_mask[:, width // 2:] & right_edges[:, width // 2:], 1, right_edges_colored[..., 0])  # Red for hippocampal edges

        axes[0, 1].imshow(left_edges_colored)
        axes[0, 1].set_title('Left Hippocampus Edges')

        axes[1, 0].imshow(right_edges_colored)
        axes[1, 0].set_title('Right Hippocampus Edges')

        # Adjust the masks to match the original image dimensions
        combined_overlay = np.copy(image)
        combined_overlay = np.stack([combined_overlay]*3, axis=-1)  # Convert to RGB

        # Apply hippocampal edges in red and other edges in blue
        combined_overlay[height // 3: 2 * height // 3, :width // 2, 0] = np.where(left_mask[height // 3: 2 * height // 3, :width // 2] & left_edges[height // 3: 2 * height // 3, :width // 2], 1, combined_overlay[height // 3: 2 * height // 3, :width // 2, 0])
        combined_overlay[height // 3: 2 * height // 3, width // 2:, 0] = np.where(right_mask[height // 3: 2 * height // 3, width // 2:] & right_edges[height // 3: 2 * height // 3, width // 2:], 1, combined_overlay[height // 3: 2 * height // 3, width // 2:, 0])

        combined_overlay[height // 3: 2 * height // 3, :width // 2, 2] = np.where(~left_mask[height // 3: 2 * height // 3, :width // 2] & left_edges[height // 3: 2 * height // 3, :width // 2], 1, combined_overlay[height // 3: 2 * height // 3, :width // 2, 2])
        combined_overlay[height // 3: 2 * height // 3, width // 2:, 2] = np.where(~right_mask[height // 3: 2 * height // 3, width // 2:] & right_edges[height // 3: 2 * height // 3, width // 2:], 1, combined_overlay[height // 3: 2 * height // 3, width // 2:, 2])

        axes[1, 1].imshow(combined_overlay)
        axes[1, 1].set_title('Hippocampal Regions Overlay')

        plt.show()
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        raise

def v8_visualize_results(image, left_edges, right_edges, left_mask, right_mask):
    try:
        height, width = image.shape

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')

        left_edges_colored = np.zeros((height, width // 2, 3), dtype=np.float32)
        right_edges_colored = np.zeros((height, width // 2, 3), dtype=np.float32)

        # Assign colors to edges and masks separately using np.where
        left_edges_colored[..., 2] = np.where(left_edges[:, :width // 2], 1, 0)  # Blue for edges
        left_edges_colored[..., 0] = np.where(left_mask[:, :width // 2] & left_edges[:, :width // 2], 1, left_edges_colored[..., 0])  # Red for hippocampal edges

        right_edges_colored[..., 2] = np.where(right_edges[:, width // 2:], 1, 0)  # Blue for edges
        right_edges_colored[..., 0] = np.where(right_mask[:, width // 2:] & right_edges[:, width // 2:], 1, right_edges_colored[..., 0])  # Red for hippocampal edges

        axes[0, 1].imshow(left_edges_colored)
        axes[0, 1].set_title('Left Hippocampus Edges')

        axes[1, 0].imshow(right_edges_colored)
        axes[1, 0].set_title('Right Hippocampus Edges')

        # Adjust the masks to match the original image dimensions
        combined_overlay = np.copy(image)
        combined_overlay = np.stack([combined_overlay]*3, axis=-1)  # Convert to RGB

        # Apply hippocampal edges in red and other edges in blue
        combined_overlay[height // 3: 2 * height // 3, width // 3: 2 * width // 3, 0] = np.where(left_mask[height // 3: 2 * height // 3, width // 3: 2 * width // 3] & left_edges[height // 3: 2 * height // 3, width // 3: 2 * width // 3], 1, combined_overlay[height // 3: 2 * height // 3, width // 3: 2 * width // 3, 0])
        combined_overlay[height // 3: 2 * height // 3, width // 3: 2 * width // 3, 0] = np.where(right_mask[height // 3: 2 * height // 3, width // 3: 2 * width // 3] & right_edges[height // 3: 2 * height // 3, width // 3: 2 * width // 3], 1, combined_overlay[height // 3: 2 * height // 3, width // 3: 2 * width // 3, 0])

        combined_overlay[height // 3: 2 * height // 3, width // 3: 2 * width // 3, 2] = np.where(~left_mask[height // 3: 2 * height // 3, width // 3: 2 * width // 3] & left_edges[height // 3: 2 * height // 3, width // 3: 2 * width // 3], 1, combined_overlay[height // 3: 2 * height // 3, width // 3: 2 * width // 3, 2])
        combined_overlay[height // 3: 2 * height // 3, width // 3: 2 * width // 3, 2] = np.where(~right_mask[height // 3: 2 * height // 3, width // 3: 2 * width // 3] & right_edges[height // 3: 2 * height // 3, width // 3: 2 * width // 3], 1, combined_overlay[height // 3: 2 * height // 3, width // 3: 2 * width // 3, 2])

        axes[1, 1].imshow(combined_overlay)
        axes[1, 1].set_title('Hippocampal Regions Overlay')

        plt.show()
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        raise

def v5_visualize_results(image, left_edges, right_edges, left_mask, right_mask):
    try:
        height, width = image.shape

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')

        left_edges_colored = np.zeros((height, width // 2, 3), dtype=np.float32)
        right_edges_colored = np.zeros((height, width // 2, 3), dtype=np.float32)

        # Assign colors to edges and masks separately using np.where
        left_edges_colored[..., 2] = np.where(left_edges[:, :width // 2], 1, 0)  # Blue for edges
        left_edges_colored[..., 0] = np.where(left_mask[:, :width // 2] & left_edges[:, :width // 2], 1, left_edges_colored[..., 0])  # Red for hippocampal edges

        right_edges_colored[..., 2] = np.where(right_edges[:, width // 2:], 1, 0)  # Blue for edges
        right_edges_colored[..., 0] = np.where(right_mask[:, width // 2:] & right_edges[:, width // 2:], 1, right_edges_colored[..., 0])  # Red for hippocampal edges

        axes[0, 1].imshow(left_edges_colored)
        axes[0, 1].set_title('Left Hippocampus Edges')

        axes[1, 0].imshow(right_edges_colored)
        axes[1, 0].set_title('Right Hippocampus Edges')

        # Adjust the masks to match the original image dimensions
        combined_overlay = np.copy(image)
        combined_overlay = np.stack([combined_overlay]*3, axis=-1)  # Convert to RGB

        # Apply hippocampal edges in red and other edges in blue
        if left_mask.any() and right_mask.any():
            combined_overlay[height // 3: 2 * height // 3, :width // 2, 0] = np.where(left_mask[height // 3: 2 * height // 3, :width // 2] & left_edges[height // 3: 2 * height // 3, :width // 2], 1, combined_overlay[height // 3: 2 * height // 3, :width // 2, 0])
            combined_overlay[height // 3: 2 * height // 3, width // 2:, 0] = np.where(right_mask[height // 3: 2 * height // 3, width // 2:] & right_edges[height // 3: 2 * height // 3, width // 2:], 1, combined_overlay[height // 3: 2 * height // 3, width // 2:, 0])

            combined_overlay[height // 3: 2 * height // 3, :width // 2, 2] = np.where(~left_mask[height // 3: 2 * height // 3, :width // 2] & left_edges[height // 3: 2 * height // 3, :width // 2], 1, combined_overlay[height // 3: 2 * height // 3, :width // 2, 2])
            combined_overlay[height // 3: 2 * height // 3, width // 2:, 2] = np.where(~right_mask[height // 3: 2 * height // 3, width // 2:] & right_edges[height // 3: 2 * height // 3, width // 2:], 1, combined_overlay[height // 3: 2 * height // 3, width // 2:, 2])
        
        axes[1, 1].imshow(combined_overlay)
        axes[1, 1].set_title('Hippocampal Regions Overlay')

        plt.show()
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        raise

def v3_visualize_results(image, left_edges, right_edges, left_mask, right_mask):
    try:
        height, width = image.shape

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')

        left_edges_colored = np.zeros((height, width // 2, 3), dtype=np.float32)
        right_edges_colored = np.zeros((height, width // 2, 3), dtype=np.float32)

        # Assign colors to edges and masks separately using np.where
        left_edges_colored[..., 2] = np.where(left_edges[:, :width // 2], 1, 0)  # Blue for edges
        left_edges_colored[..., 0] = np.where(left_mask[:, :width // 2] & left_edges[:, :width // 2], 1, left_edges_colored[..., 0])  # Red for hippocampal edges

        right_edges_colored[..., 2] = np.where(right_edges[:, width // 2:], 1, 0)  # Blue for edges
        right_edges_colored[..., 0] = np.where(right_mask[:, width // 2:] & right_edges[:, width // 2:], 1, right_edges_colored[..., 0])  # Red for hippocampal edges

        axes[0, 1].imshow(left_edges_colored)
        axes[0, 1].set_title('Left Hippocampus Edges')

        axes[1, 0].imshow(right_edges_colored)
        axes[1, 0].set_title('Right Hippocampus Edges')

        # Adjust the masks to match the original image dimensions
        combined_overlay = np.copy(image)
        combined_overlay = np.stack([combined_overlay]*3, axis=-1)  # Convert to RGB

        # Apply hippocampal edges in red and other edges in blue
        combined_overlay[height // 3: 2 * height // 3, :width // 2, 0] = np.where(left_mask[height // 3: 2 * height // 3, :width // 2] & left_edges[height // 3: 2 * height // 3, :width // 2], 1, combined_overlay[height // 3: 2 * height // 3, :width // 2, 0])
        combined_overlay[height // 3: 2 * height // 3, width // 2:, 0] = np.where(right_mask[height // 3: 2 * height // 3, width // 2:] & right_edges[height // 3: 2 * height // 3, width // 2:], 1, combined_overlay[height // 3: 2 * height // 3, width // 2:, 0])

        combined_overlay[height // 3: 2 * height // 3, :width // 2, 2] = np.where(~left_mask[height // 3: 2 * height // 3, :width // 2] & left_edges[height // 3: 2 * height // 3, :width // 2], 1, combined_overlay[height // 3: 2 * height // 3, :width // 2, 2])
        combined_overlay[height // 3: 2 * height // 3, width // 2:, 2] = np.where(~right_mask[height // 3: 2 * height // 3, width // 2:] & right_edges[height // 3: 2 * height // 3, width // 2:], 1, combined_overlay[height // 3: 2 * height // 3, width // 2:, 2])

        axes[1, 1].imshow(combined_overlay)
        axes[1, 1].set_title('Hippocampal Regions Overlay')

        plt.show()
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        raise


def v2_visualize_results(image, left_edges, right_edges, left_mask, right_mask):
    try:
        height, width = image.shape

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')

        left_edges_colored = np.zeros((height, width // 2, 3), dtype=np.float32)
        right_edges_colored = np.zeros((height, width // 2, 3), dtype=np.float32)

        # Assign colors to edges and masks separately using np.where
        left_edges_colored[..., 2] = np.where(left_edges[:, :width // 2], 1, 0)  # Blue for edges
        left_edges_colored[..., 0] = np.where(left_mask[:, :width // 2] & left_edges[:, :width // 2], 1, left_edges_colored[..., 0])  # Red for hippocampal edges

        right_edges_colored[..., 2] = np.where(right_edges[:, width // 2:], 1, 0)  # Blue for edges
        right_edges_colored[..., 0] = np.where(right_mask[:, width // 2:] & right_edges[:, width // 2:], 1, right_edges_colored[..., 0])  # Red for hippocampal edges

        axes[0, 1].imshow(left_edges_colored)
        axes[0, 1].set_title('Left Hippocampus Edges')

        axes[1, 0].imshow(right_edges_colored)
        axes[1, 0].set_title('Right Hippocampus Edges')

        # Adjust the masks to match the original image dimensions
        combined_overlay = np.copy(image)
        combined_overlay = np.stack([combined_overlay]*3, axis=-1)  # Convert to RGB

        # Apply hippocampal edges in red and other edges in blue
        combined_overlay[height // 3: 2 * height // 3, :width // 2, 0] = np.where(left_mask[height // 3: 2 * height // 3, :width // 2] & left_edges[height // 3: 2 * height // 3, :width // 2], 1, combined_overlay[height // 3: 2 * height // 3, :width // 2, 0])
        combined_overlay[height // 3: 2 * height // 3, width // 2:, 0] = np.where(right_mask[height // 3: 2 * height // 3, width // 2:] & right_edges[height // 3: 2 * height // 3, width // 2:], 1, combined_overlay[height // 3: 2 * height // 3, width // 2:, 0])

        combined_overlay[height // 3: 2 * height // 3, :width // 2, 2] = np.where(~left_mask[height // 3: 2 * height // 3, :width // 2] & left_edges[height // 3: 2 * height // 3, :width // 2], 1, combined_overlay[height // 3: 2 * height // 3, :width // 2, 2])
        combined_overlay[height // 3: 2 * height // 3, width // 2:, 2] = np.where(~right_mask[height // 3: 2 * height // 3, width // 2:] & right_edges[height // 3: 2 * height // 3, width // 2:], 1, combined_overlay[height // 3: 2 * height // 3, width // 2:, 2])

        axes[1, 1].imshow(combined_overlay)
        axes[1, 1].set_title('Hippocampal Regions Overlay')

        plt.show()
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        raise

def old_visualize_results(image, left_edges, right_edges, left_mask, right_mask):
    try:
        height, width = image.shape

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')

        left_edges_colored = np.zeros((height, width // 2, 3), dtype=np.float32)
        right_edges_colored = np.zeros((height, width // 2, 3), dtype=np.float32)

        # Assign colors to edges and masks separately using np.where
        left_edges_colored[..., 2] = np.where(left_edges[:, :width // 2], 1, 0)  # Blue for edges
        left_edges_colored[..., 0] = np.where(left_mask[:, :width // 2] & left_edges[:, :width // 2], 1, left_edges_colored[..., 0])  # Red for hippocampal edges

        right_edges_colored[..., 2] = np.where(right_edges[:, width // 2:], 1, 0)  # Blue for edges
        right_edges_colored[..., 0] = np.where(right_mask[:, width // 2:] & right_edges[:, width // 2:], 1, right_edges_colored[..., 0])  # Red for hippocampal edges

        axes[0, 1].imshow(left_edges_colored)
        axes[0, 1].set_title('Left Hippocampus Edges')

        axes[1, 0].imshow(right_edges_colored)
        axes[1, 0].set_title('Right Hippocampus Edges')

        # Adjust the masks to match the original image dimensions
        combined_overlay = np.copy(image)
        combined_overlay = np.stack([combined_overlay]*3, axis=-1)  # Convert to RGB

        # Apply hippocampal edges in red and other edges in blue
        combined_overlay[height // 3: 2 * height // 3, :width // 2, 0] = np.where(left_mask[height // 3: 2 * height // 3, :width // 2] & left_edges[height // 3: 2 * height // 3, :width // 2], 1, combined_overlay[height // 3: 2 * height // 3, :width // 2, 0])
        combined_overlay[height // 3: 2 * height // 3, width // 2:, 0] = np.where(right_mask[height // 3: 2 * height // 3, width // 2:] & right_edges[height // 3: 2 * height // 3, width // 2:], 1, combined_overlay[height // 3: 2 * height // 3, width // 2:, 0])

        combined_overlay[height // 3: 2 * height // 3, :width // 2, 2] = np.where(~left_mask[height // 3: 2 * height // 3, :width // 2] & left_edges[height // 3: 2 * height // 3, :width // 2], 1, combined_overlay[height // 3: 2 * height // 3, :width // 2, 2])
        combined_overlay[height // 3: 2 * height // 3, width // 2:, 2] = np.where(~right_mask[height // 3: 2 * height // 3, width // 2:] & right_edges[height // 3: 2 * height // 3, width // 2:], 1, combined_overlay[height // 3: 2 * height // 3, width // 2:, 2])

        axes[1, 1].imshow(combined_overlay)
        axes[1, 1].set_title('Hippocampal Regions Overlay')

        plt.show()
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        raise

def old_visualize_results(image, left_edges, right_edges, left_mask, right_mask):
    try:
        height, width = image.shape

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')

        left_edges_colored = np.zeros((height, width // 2, 3), dtype=np.float32)
        right_edges_colored = np.zeros((height, width // 2, 3), dtype=np.float32)

        # Assign colors to edges and masks separately using np.where
        left_edges_colored[..., 2] = np.where(left_edges[:, :width // 2], 1, 0)  # Blue for edges
        left_edges_colored[..., 0] = np.where(left_mask[:, :width // 2] & left_edges[:, :width // 2], 1, left_edges_colored[..., 0])  # Red for hippocampal edges

        right_edges_colored[..., 2] = np.where(right_edges[:, width // 2:], 1, 0)  # Blue for edges
        right_edges_colored[..., 0] = np.where(right_mask[:, width // 2:] & right_edges[:, width // 2:], 1, right_edges_colored[..., 0])  # Red for hippocampal edges

        axes[0, 1].imshow(left_edges_colored)
        axes[0, 1].set_title('Left Hippocampus Edges')

        axes[1, 0].imshow(right_edges_colored)
        axes[1, 0].set_title('Right Hippocampus Edges')

        # Adjust the masks to match the original image dimensions
        combined_overlay = np.copy(image)
        combined_overlay = np.stack([combined_overlay]*3, axis=-1)  # Convert to RGB

        # Apply hippocampal edges in red and other edges in blue
        combined_overlay[height // 3: 2 * height // 3, :width // 2, 0] = np.where(left_mask[height // 3: 2 * height // 3, :width // 2] & left_edges[height // 3: 2 * height // 3, :width // 2], 1, combined_overlay[height // 3: 2 * height // 3, :width // 2, 0])
        combined_overlay[height // 3: 2 * height // 3, width // 2:, 0] = np.where(right_mask[height // 3: 2 * height // 3, width // 2:] & right_edges[height // 3: 2 * height // 3, width // 2:], 1, combined_overlay[height // 3: 2 * height // 3, width // 2:, 0])

        combined_overlay[height // 3: 2 * height // 3, :width // 2, 2] = np.where(~left_mask[height // 3: 2 * height // 3, :width // 2] & left_edges[height // 3: 2 * height // 3, :width // 2], 1, combined_overlay[height // 3: 2 * height // 3, :width // 2, 2])
        combined_overlay[height // 3: 2 * height // 3, width // 2:, 2] = np.where(~right_mask[height // 3: 2 * height // 3, width // 2:] & right_edges[height // 3: 2 * height // 3, width // 2:], 1, combined_overlay[height // 3: 2 * height // 3, width // 2:, 2])

        axes[1, 1].imshow(combined_overlay)
        axes[1, 1].set_title('Hippocampal Regions Overlay')

        plt.show()
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        raise

class TestFunctions(unittest.TestCase):
    def setUp(self):
        self.height = 512
        self.width = 512
        self.image = np.random.rand(self.height, self.width).astype(np.float32)

        self.left_edges = np.zeros_like(self.image)
        self.right_edges = np.zeros_like(self.image)
        self.left_edges[200:300, 100:200] = 1
        self.right_edges[200:300, 300:400] = 1

        self.left_mask = np.zeros_like(self.image)
        self.right_mask = np.zeros_like

        self.right_mask = np.zeros_like(self.image)
        self.left_mask[220:280, 120:180] = 1
        self.right_mask[220:280, 320:380] = 1

    def test_load_image(self):
        # This test would normally involve loading an actual image file,
        # but for simplicity, we'll just check if the function handles a valid path
        # and raises an error for an invalid path.
        valid_path = 'valid_image_path.png'
        invalid_path = 'invalid_image_path.png'
        with self.assertRaises(ValueError):
            load_image(invalid_path)
        # Assuming valid_path exists for the actual test scenario

    def test_preprocess_image(self):
        preprocessed_image = preprocess_image(self.image)
        self.assertEqual(preprocessed_image.shape, self.image.shape)
        self.assertTrue((preprocessed_image >= 0).all() and (preprocessed_image <= 1).all())

    def test_skull_stripping(self):
        brain_image = skull_stripping(self.image)
        self.assertEqual(brain_image.shape, self.image.shape)
        self.assertTrue((brain_image >= 0).all() and (brain_image <= 1).all())

    def test_edge_detection(self):
        edges = edge_detection(self.image)
        self.assertEqual(edges.shape, self.image.shape)
        self.assertTrue((edges >= 0).all() and (edges <= 1).all())

    def test_extract_hippocampal_region_with_detection(self):
        hippocampal_mask = extract_hippocampal_region(self.left_edges, side='left')
        self.assertEqual(hippocampal_mask.shape, self.image.shape)
        self.assertTrue((hippocampal_mask == 0).sum() + (hippocampal_mask == 1).sum() == self.image.size)
        
        hippocampal_mask = extract_hippocampal_region(self.right_edges, side='right')
        self.assertEqual(hippocampal_mask.shape, self.image.shape)
        self.assertTrue((hippocampal_mask == 0).sum() + (hippocampal_mask == 1).sum() == self.image.size)

    def test_extract_hippocampal_region_without_detection(self):
        empty_edges = np.zeros_like(self.image)
        hippocampal_mask = extract_hippocampal_region(empty_edges, side='left')
        self.assertEqual(hippocampal_mask.shape, self.image.shape)
        self.assertTrue((hippocampal_mask == 0).all())
        
        hippocampal_mask = extract_hippocampal_region(empty_edges, side='right')
        self.assertEqual(hippocampal_mask.shape, self.image.shape)
        self.assertTrue((hippocampal_mask == 0).all())

    def test_compare_hippocampal_regions(self):
        left_area, right_area = compare_hippocampal_regions(self.left_mask, self.right_mask)
        self.assertEqual(left_area, (self.left_mask > 0).sum())
        self.assertEqual(right_area, (self.right_mask > 0).sum())
        
        left_area, right_area = compare_hippocampal_regions(np.zeros_like(self.left_mask), np.zeros_like(self.right_mask))
        self.assertEqual(left_area, 0)
        self.assertEqual(right_area, 0)

    def test_visualize_results(self):
        # This test is mainly to ensure the function runs without error.
        # Visualization results should be checked manually or with image comparison tools.
        try:
            visualize_results(self.image, self.left_edges, self.right_edges, self.left_mask, self.right_mask)
        except Exception as e:
            self.fail(f"visualize_results raised an exception: {e}")

def main():
    try:
        #        unittest.main(argv=[''], exit=False)

        # Path to the image file
        image_file_path = '/Users/davidbrewster/Downloads/output/1035_MR_20240620_123527.864401/frame_0063.png'
        
        # Load and preprocess image
        image = load_image(image_file_path)
        preprocessed_image = preprocess_image(image)
        
        # Skull stripping
        brain_image = skull_stripping(preprocessed_image)
        
        # Edge detection
        edges = edge_detection(brain_image)
        
        # Extract hippocampal regions
        left_mask = extract_hippocampal_region(brain_image, edges, side='left')
        right_mask = extract_hippocampal_region(brain_image, edges, side='right')
        
        # Compare hippocampal regions
        left_area, right_area = compare_hippocampal_regions(left_mask, right_mask)
        
        # Visualize results
        visualize_results(brain_image, edges[:, :edges.shape[1] // 2], edges[:, edges.shape[1] // 2:], left_mask, right_mask)

    except Exception as e:
        logging.critical(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()

