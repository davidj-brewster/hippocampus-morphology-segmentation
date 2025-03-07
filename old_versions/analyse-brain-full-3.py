import logging
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, filters, measure
import cv2
from PIL import Image


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

def frequency_analysis(image):
    """
    Perform frequency analysis on the brain image.

    Parameters:
    image (numpy.ndarray): Input brain image.

    Returns:
    numpy.ndarray: Frequency spectrum of the image.
    """
    try:
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift))
        return magnitude_spectrum
    except Exception as e:
        logging.error(f"Error during frequency analysis: {e}")
        raise


def largest_connected_component(binary_mask):
    labeled_mask, num_labels = measure.label(binary_mask, connectivity=2, return_num=True)
    if num_labels == 0:
        return binary_mask
    largest_label = 1 + np.argmax(np.bincount(labeled_mask.flat)[1:])
    return labeled_mask == largest_label


def detect_density_anomalies(mask, skull_mask, threshold=0.5):
    try:
        mask = mask.astype(bool)
        skull_mask = skull_mask.astype(bool)
        combined_mask = measure.label(mask & skull_mask)
        labeled_mask = measure.label(combined_mask)
        regions = measure.regionprops(labeled_mask)
        areas = [r.area for r in regions]
        mean_area = np.mean(areas)
        anomalies = np.zeros_like(mask, dtype=bool)
        for region in regions:
            if abs(region.area - mean_area) / mean_area > threshold:
                anomalies[tuple(region.coords.T)] = True
                logging.info(f"{tuple(region.coords.T)}")
        return anomalies
    except Exception as e:
        logging.error(f"Error during frequency analysis: {e}")
        raise



def detect_bright_spots(image, mask, skull_mask, threshold=0.95):
    try:
        mask = mask.astype(bool)
        skull_mask = skull_mask.astype(bool)
        bright_spots = (image > threshold) & mask & skull_mask
        return bright_spots
    except Exception as e:
        logging.error(f"Error detecting bright spots: {e}")
        raise

def detect_dark_spots(image, mask, skull_mask, threshold=0.05):
    try:
        mask = mask.astype(bool)
        skull_mask = skull_mask.astype(bool)
        dark_spots = (image < threshold) & mask & skull_mask
        return dark_spots
    except Exception as e:
        logging.error(f"Error detecting dark spots: {e}")
        raise

def detect_outliers(mask, skull_mask, threshold=0.1):
    try:
        mask = mask.astype(bool)
        skull_mask = skull_mask.astype(bool)
        combined_mask = measure.label(mask & skull_mask)
        labeled_mask = measure.label(combined_mask)
        regions = measure.regionprops(labeled_mask)

        areas = [r.area for r in regions]
        mean_area = np.mean(areas)
        anomalies = np.zeros_like(mask, dtype=bool)
        for region in regions:
            if abs(region.area - mean_area) / mean_area > threshold:
                anomalies[tuple(region.coords.T)] = True
                logging.info(f"{tuple(region.coords.T)}")
        return anomalies
    except Exception as e:
        logging.error(f"Error detecting outliers: {e}")
        raise


def detect_hippocampus_anomalies(left_mask, right_mask, skull_mask):
    try:
        combined_mask = left_mask.astype(bool) | right_mask.astype(bool)
        anomalies = {
            'density_anomalies': detect_density_anomalies(combined_mask, skull_mask),
            'bright_spots': detect_bright_spots(combined_mask, combined_mask, skull_mask),
            'dark_spots': detect_dark_spots(combined_mask, combined_mask, skull_mask),
            'outliers': detect_outliers(combined_mask, skull_mask)
        }
        return anomalies
    except Exception as e:
        logging.error(f"Error detecting hippocampus anomalies: {e}")
        raise


def vold_detect_density_anomalies(mask, skull_mask, threshold=3):
    """
    Detects regions with density anomalies within the hippocampal mask.
    
    Parameters:
    mask (np.ndarray): Binary mask of the region of interest.
    skull_mask (np.ndarray): Binary mask of the skull-stripped brain.
    threshold (float): Threshold for detecting anomalies based on area deviation.
    
    Returns:
    np.ndarray: Binary mask of the detected anomalies.
    """
    try:
        logging.info("Starting density anomaly detection")
        mask = mask.astype(bool)
        skull_mask = skull_mask.astype(bool)
        labeled_mask = measure.label(mask & skull_mask)
        regions = measure.regionprops(labeled_mask)
        areas = [r.area for r in regions]
        mean_area = np.mean(areas)
        anomalies = np.zeros_like(mask, dtype=bool)
        for region in regions:
            if abs(region.area - mean_area) / mean_area > threshold:
                anomalies[tuple(region.coords.T)] = True
        logging.info("Density anomaly detection completed")
        return anomalies
    except Exception as e:
        logging.error(f"Error detecting density anomalies: {e}")
        raise



def skull_stripping(image, output_path="/tmp", min_size=500, area_threshold=5000):
    """
    Perform skull stripping on the brain MRI image.

    Parameters:
    image (np.ndarray): Input brain image.
    output_path (str): Path to save the visual representation of skull stripping.
    min_size (int): Minimum size for small object removal.
    area_threshold (int): Area threshold for small hole removal.

    Returns:
    tuple: Skull stripped image and brain mask.
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


def edge_detection(image, output_path='/tmp/', threshold=0.05):
    try:
        logging.info("Starting edge detection")
        edges = filters.sobel(image)
        edges = morphology.binary_dilation(edges > threshold)
        edge_coords = np.nonzero(edges)
        logging.info(f"Edge coordinates detected with threshold {threshold}: {len(edge_coords[0])} edges")
        
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


def new_region_of_interest(edges, region):
    height, width = edges.shape
    if region == 'left_hippocampus':
        return edges[height // 3: 2  * height // 3, width // 3: width // 2]
    elif region == 'right_hippocampus':
        return edges[height // 3: 2 * height // 3, width // 2: 2 * width // 3]
    elif region == 'temporal_lobe':
        return edges[height // 4: 3 * height // 4, width // 4: 3 * width // 4]
    else:
        raise ValueError("Invalid region specified")


def region_of_interest(brain_image, edges, region):
    """
    Extract the region of interest from the brain image based on the specified region.

    Parameters:
    brain_image (np.ndarray): Input brain image.
    edges (np.ndarray): Binary mask of detected edges.
    region (str): The region of interest to extract ('left_hippocampus', 'right_hippocampus', 'temporal_lobe').

    Returns:
    np.ndarray: Extracted region of interest from the edges mask.
    """
    try:
        logging.info(f"Extracting region of interest: {region}")
        height, width = brain_image.shape
        logging.info(f"Image dimensions: {height} x {width}")

        if region == 'left_hippocampus':
            roi = edges[height // 3: 2 ** height // 3, width // 2 - width // 12 : width // 2]
        elif region == 'right_hippocampus':
            roi = edges[height // 3: 2 ** height // 3, width // 2: width // 2 + width // 12]
        elif region == 'temporal_lobe':
            roi = edges[6 * height // 8: height, width // 3: 2 * width // 3]
        else:
            raise ValueError("Invalid region specified")

        logging.info(f"Region of interest '{region}' extracted with shape: {roi.shape}")
        return roi
    except Exception as e:
        logging.error(f"Error extracting region of interest: {e}")
        raise


def extract_region(brain_image, edges, region, min_area=500, max_area=4000):
    try:
        logging.info(f"Extracting {region}")
        height, width = brain_image.shape

        roi = region_of_interest(brain_image, edges, region)
        roi_edge_coords = np.nonzero(roi)
        logging.info(f"Edges coordinates considered for {region}: {len(roi_edge_coords[0])} edges")

        labeled_roi = measure.label(roi)
        regions = measure.regionprops(labeled_roi)
        filtered_regions = [r for r in regions if min_area < r.area < max_area]
        logging.info(f"{np.nonzero(filtered_regions)}")

        region_mask = np.zeros_like(brain_image, dtype=bool)
        if filtered_regions:
            largest_region = max(filtered_regions, key=lambda r: r.area)
            for coord in largest_region.coords:
                y, x = coord
                if region == 'left_hippocampus':
                    #region_mask[y + brain_image.shape[0] // 3, x + brain_image.shape[1] // 2] = True
                    region_mask[y + brain_image.shape[0] // 3, x + width // 2 - width // 12] = True
                elif region == 'right_hippocampus':
                    region_mask[y + brain_image.shape[0] // 3, x + brain_image.shape[1] // 2] = True

        selected_edge_coords = np.nonzero(region_mask)
        logging.info(f"Selected edge coordinates for {region}: {len(selected_edge_coords[0])} edges")

        logging.info(f"{region.capitalize()} extracted with area: {np.sum(region_mask)}")
        return region_mask
    except Exception as e:
        logging.error(f"Error extracting {region}: {e}")
        raise

def v4_extract_region(brain_image, edges, region, min_area=1000, max_area=5000):
    try:
        logging.info(f"Extracting {region}")
        roi = region_of_interest(brain_image, edges, region)
        roi_edge_coords = np.nonzero(roi)
        logging.info(f"Edges coordinates considered for {region}: {len(roi_edge_coords[0])} edges")

        labeled_roi = measure.label(roi)
        regions = measure.regionprops(labeled_roi)
        filtered_regions = [r for r in regions if min_area < r.area < max_area]

        region_mask = np.zeros_like(brain_image, dtype=np.bool_)
        if filtered_regions:
            largest_region = max(filtered_regions, key=lambda r: r.area)
            for coord in largest_region.coords:
                y, x = coord
                # Adjust coordinates to match the original image dimensions
                if region == 'left_hippocampus':
                    region_mask[y + brain_image.shape[0] // 3, x + brain_image.shape[1] // 2 - brain_image.shape[1] // 12] = True
                elif region == 'right_hippocampus':
                    region_mask[y + brain_image.shape[0] // 3, x + brain_image.shape[1] // 2] = True

        selected_edge_coords = np.nonzero(region_mask)
        logging.info(f"Selected edge coordinates for {region}: {len(selected_edge_coords[0])} edges")

        logging.info(f"{region.capitalize()} extracted with area: {np.sum(region_mask)}")
        return region_mask
    except Exception as e:
        logging.error(f"Error extracting {region}: {e}")
        raise


def old_detect_hippocampus_anomalies(left_hippocampus, right_hippocampus, intensity_threshold=2, density_threshold=0.2):
    """
    Detect anomalies in the hippocampus regions by comparing left and right hippocampus.

    Parameters:
    left_hippocampus (numpy.ndarray): Left hippocampus region.
    right_hippocampus (numpy.ndarray): Right hippocampus region.
    intensity_threshold (float): Z-score threshold for intensity anomalies.
    density_threshold (float): Proportional difference threshold for density anomalies.

    Returns:
    dict: Dictionary containing binary masks for detected anomalies.
    """
    try:
        intensity_anomalies_left = detect_intensity_anomalies(left_hippocampus, threshold=intensity_threshold)
        intensity_anomalies_right = detect_intensity_anomalies(right_hippocampus, threshold=intensity_threshold)
        density_anomalies = detect_density_anomalies(left_hippocampus, right_hippocampus, threshold=density_threshold)
        
        return {
            'intensity_anomalies_left': intensity_anomalies_left,
            'intensity_anomalies_right': intensity_anomalies_right,
            'density_anomalies': density_anomalies
        }
    except Exception as e:
        logging.error(f"Error during hippocampus anomaly detection: {e}")
        raise


def visualize_results(image, regions, region_masks, anomalies=None):
    try:
        height, width = image.shape
        logging.info(f"Image dimensions: {height}, {width}")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10),linewidth=1)
        
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        
        for i, region in enumerate(regions):
            mask = region_masks[region]
            #overlay = np.zeros((height, width, 3), dtype=np.float32)
            overlay = np.stack([image]*3, axis=-1)  # Convert to RGB
            overlay[..., 2] = np.where(edge_detection(image), 1, 0)  # Blue for edges
            overlay[..., 0] = np.where(mask & edge_detection(image), 1, 0)  # Red for region edges
            
            if anomalies and region in anomalies:
                anomaly = anomalies[region]
                if 'density_anomalies' in anomaly and anomaly['density_anomalies']:
                    overlay[..., 1] = np.where(mask, 1, 0)  # Green for density anomalies
            
            ax = axes[i // 3, i % 2]
            ax.imshow(overlay)
            ax.set_title(f'{region.replace("_", " ").capitalize()} Edges and Anomalies')

        # Display the original image in the bottom quadrants as well
        axes[1, 0].imshow(image, cmap='gray')
        axes[1, 0].set_title('Original Image (Left)')

        axes[1, 1].imshow(image, cmap='gray')
        axes[1, 1].set_title('Original Image (Right)')

        plt.rcParams.update({'legend.fontsize': 0.1})
        plt.savefig("output_visualization.png")
        plt.show()
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        raise

def load_image(image_path):
    """Load an image from a file path and convert to grayscale numpy array."""
    try:
        image = Image.open(image_path).convert('L')
        return np.array(image) / 255.0
    except Exception as e:
        logging.error(f"Error loading image: {e}")
        raise

def main(input_file):
    logging.basicConfig(level=logging.INFO)

    try:
        # Load the image
        image = load_image(input_file)
        logging.info(f"Loaded image: {input_file}")

        # Define the output path base
        output_path_base = input_file.replace('.png', '')

        # Perform skull stripping
        skull_stripped_image, skull_mask = skull_stripping(image, output_path_base)
        logging.info(f"Performed skull stripping on: {input_file}")

        # Perform edge detection
        edges = edge_detection(skull_stripped_image, output_path_base)
        logging.info(f"Performed edge detection on: {input_file}")

        # Extract hippocampal regions
        left_mask = extract_region(skull_stripped_image, edges, 'left_hippocampus')
        right_mask = extract_region(skull_stripped_image, edges, 'right_hippocampus')

        # Convert masks to boolean type if not already
        left_mask = left_mask.astype(bool)
        right_mask = right_mask.astype(bool)
        skull_mask = skull_mask.astype(bool)

        # Detect hippocampal anomalies
        anomalies = detect_hippocampus_anomalies(left_mask, right_mask, skull_mask)
        logging.info(f"Detected anomalies in hippocampal regions")

        # Visualize results
        visualize_results(skull_stripped_image, ['left_hippocampus', 'right_hippocampus'], 
                          {'left_hippocampus': left_mask, 'right_hippocampus': right_mask}, anomalies)
        logging.info(f"Visualization completed for: {input_file}")

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Brain MRI analysis")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input image file')
    
    args = parser.parse_args()
    main(args.input_file)

