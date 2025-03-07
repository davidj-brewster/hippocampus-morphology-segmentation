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

def detect_density_anomalies(mask, skull_mask, threshold=0.5):
    try:
        regions = measure.regionprops(mask & skull_mask)
        areas = [r.area for r in regions]
        mean_area = np.mean(areas)
        anomalies = np.zeros_like(mask, dtype=bool)
        for region in regions:
            if abs(region.area - mean_area) / mean_area > threshold:
                anomalies[tuple(region.coords.T)] = True
        return anomalies
    except Exception as e:
        logging.error(f"Error detecting density anomalies: {e}")
        raise

def detect_bright_spots(image, mask, skull_mask, threshold=0.95):
    try:
        bright_spots = (image > threshold) & mask & skull_mask
        return bright_spots
    except Exception as e:
        logging.error(f"Error detecting bright spots: {e}")
        raise

def detect_dark_spots(image, mask, skull_mask, threshold=0.05):
    try:
        dark_spots = (image < threshold) & mask & skull_mask
        return dark_spots
    except Exception as e:
        logging.error(f"Error detecting dark spots: {e}")
        raise

def detect_outliers(mask, skull_mask, threshold=0.1):
    try:
        regions = measure.regionprops(mask & skull_mask)
        areas = [r.area for r in regions]
        mean_area = np.mean(areas)
        anomalies = np.zeros_like(mask, dtype=bool)
        for region in regions:
            if abs(region.area - mean_area) / mean_area > threshold:
                anomalies[tuple(region.coords.T)] = True
        return anomalies
    except Exception as e:
        logging.error(f"Error detecting outliers: {e}")
        raise

def detect_hippocampus_anomalies(left_mask, right_mask, skull_mask):
    try:
        combined_mask = left_mask | right_mask
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

def v1_detect_intensity_anomalies(image, threshold=2): 
    """
    Detect intensity anomalies in the brain image.

    Parameters:
    image (numpy.ndarray): Input brain image.
    threshold (float): Z-score threshold for detecting anomalies.

    Returns:
    numpy.ndarray: Binary mask of intensity anomalies.
    """
    try:
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        z_scores = (image - mean_intensity) / std_intensity
        anomalies = np.abs(z_scores) > threshold

        return anomalies
    except Exception as e:
        logging.error(f"Error during intensity anomaly detection: {e}")
        raise

def v2_detect_density_anomalies(left_region, right_region, threshold=0.5):
    """
    Detect density anomalies between two brain regions.

    Parameters:
    left_region (numpy.ndarray): Left brain region.
    right_region (numpy.ndarray): Right brain region.
    threshold (float): Proportional difference threshold for detecting anomalies.

    Returns:
    bool: True if a density anomaly is detected, False otherwise.
    """
    try:
        left_density = np.mean(left_region)
        right_density = np.mean(right_region)
        density_diff = np.abs(left_density - right_density) / max(left_density, right_density)
        return density_diff > threshold
    except Exception as e:
        logging.error(f"Error during density anomaly detection: {e}")
        raise

def v2_detect_bright_spots(image, threshold=0.98):
    """
    Detect bright spots in the brain image.

    Parameters:
    image (numpy.ndarray): Input brain image.
    threshold (float): Intensity threshold for detecting bright spots.

    Returns:
    numpy.ndarray: Binary mask of bright spots.
    """
    try:
        bright_spots = image > threshold
        return bright_spots
    except Exception as e:
        logging.error(f"Error during bright spot detection: {e}")
        raise

def v2_detect_dark_spots(image, threshold=0.1):
    """
    Detect dark spots in the brain image.

    Parameters:
    image (numpy.ndarray): Input brain image.
    threshold (float): Intensity threshold for detecting dark spots.

    Returns:
    numpy.ndarray: Binary mask of dark spots.
    """
    try:
        dark_spots = image < threshold
        return dark_spots
    except Exception as e:
        logging.error(f"Error during dark spot detection: {e}")
        raise

def v2_detect_hippocampus_anomalies(brain_image,left_hippocampus, right_hippocampus, intensity_threshold=15, density_threshold=15):
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
        intensity_anomalies_left = detect_intensity_anomalies(brain_image, threshold=intensity_threshold)
        intensity_anomalies_right = detect_intensity_anomalies(brain_image, threshold=intensity_threshold)
        density_anomalies = detect_density_anomalies(left_hippocampus, right_hippocampus, threshold=density_threshold)
        dark_spots = detect_dark_spots(left_hippocampus)
        bright_spots = detect_bright_spots(left_hippocampus) 
        return {
            'intensity_anomalies_left': intensity_anomalies_left,
            'intensity_anomalies_right': intensity_anomalies_right,
            'density_anomalies': density_anomalies,
            'detect_dark_spots': dark_spots,
            'detect_bright_spots': bright_spots
        }
    except Exception as e:
        logging.error(f"Error during hippocampus anomaly detection: {e}")
        raise


def largest_connected_component(binary_mask):
    labeled_mask, num_labels = measure.label(binary_mask, connectivity=2, return_num=True)
    if num_labels == 0:
        return binary_mask
    largest_label = 1 + np.argmax(np.bincount(labeled_mask.flat)[1:])
    return labeled_mask == largest_label

def skull_stripping(image, output_path='/tmp/', min_size=100, area_threshold=5000):
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
        return brain_extracted
    except Exception as e:
        logging.error(f"Error during skull stripping: {e}")
        raise

def edge_detection(image, output_path='/tmp/', threshold=0.115):
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


def region_of_interest(edges, region):
    height, width = edges.shape
    if region == 'left_hippocampus':
        return edges[height // 3: 2 * height // 3, width // 3: width // 2]
    elif region == 'right_hippocampus':
        return edges[height // 3: 2 * height // 3, width // 2: 2 * width // 3]
    elif region == 'temporal_lobe':
        return edges[height // 4: 3 * height // 4, width // 4: 3 * width // 4]
    else:
        raise ValueError("Invalid region specified")


def old_region_of_interest(brain_image, edges, region):
    logging.info(f"{brain_image.shape} {edges.shape}")
    height, width = brain_image.shape
    if region == 'left_hippocampus':
        return edges[height // 3: 2 * height // 3, width // 2 - width // 12 : width // 2]
    elif region == 'right_hippocampus':
        return edges[height // 3: 2 * height // 3, width // 2: width // 2 + width // 12]
    elif region == 'temporal_lobe':
        return edges[7*height // 8: height, :width ]
    else:
        raise ValueError("Invalid region specified")

def extract_region(brain_image, edges, region, min_area=200, max_area=80000):
    try:
        logging.info(f"Extracting {region}")
        roi = region_of_interest(edges, region)
        roi_edge_coords = np.nonzero(roi)
        logging.info(f"Edges coordinates considered for {region}: {len(roi_edge_coords[0])} edges")

        labeled_roi = measure.label(roi)
        regions = measure.regionprops(labeled_roi)
        filtered_regions = [r for r in regions if min_area < r.area < max_area]

        region_mask = np.zeros_like(brain_image, dtype=bool)
        if filtered_regions:
            largest_region = max(filtered_regions, key=lambda r: r.area)
            for coord in largest_region.coords:
                y, x = coord
                if region == 'left_hippocampus':
                    region_mask[y + brain_image.shape[0] // 3, x + brain_image.shape[1] // 3] = True
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


#def detect_hippocampus_anomalies(left_mask, right_mask, threshold=0.1):
    try:
        left_area = np.sum(left_mask)
        right_area = np.sum(right_mask)
        density_diff = np.abs(left_area - right_area) / max(left_area, right_area)
        anomalies = {'density_anomalies': density_diff > threshold}
        return anomalies
    except Exception as e:
        logging.error(f"Error detecting hippocampus anomalies: {e}")
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

    # Load the image
    image = load_image(input_file)

    # Define the output path base
    output_path_base = input_file.replace('.png', '')

    image = preprocess_image(image)

    # Perform skull stripping
    skull_stripped_image = skull_stripping(image, output_path_base)

    # Perform edge detection
    edges = edge_detection(skull_stripped_image, output_path_base)

    # Extract hippocampal regions
    #left_mask = extract_region(skull_stripped_image, edges, 'left_hippocampus')
    #right_mask = extract_region(skull_stripped_image, edges, 'right_hippocampus')

    # Detect hippocampal anomalies
    #anomalies = detect_hippocampus_anomalies(image, left_mask, right_mask)
    ##anomalies = detect_intensity_anomalies(left_mask, right_mask)
    # Visualize results
    #visualize_results(image, ['left_hippocampus', 'right_hippocampus'], 
    #                  {'left_hippocampus': left_mask, 'right_hippocampus': right_mask}, anomalies)

    # Extract hippocampal regions
    left_mask = extract_region(skull_stripped_image, edges, 'left_hippocampus')
    right_mask = extract_region(skull_stripped_image, edges, 'right_hippocampus')
    skull_mask = extract_region(skull_stripped_image, edges, 'temporal_lobe')
    # Convert masks to boolean type if not already
    left_mask = left_mask.astype(bool)
    right_mask = right_mask.astype(bool)
    skull_mask = skull_mask.astype(bool)

    # Detect hippocampal anomalies
    anomalies = detect_hippocampus_anomalies(left_mask, right_mask, skull_mask)

    # Visualize results
    visualize_results(skull_stripped_image, ['left_hippocampus', 'right_hippocampus'], 
                      {'left_hippocampus': left_mask, 'right_hippocampus': right_mask}, anomalies)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Brain MRI analysis")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input image file')
    
    args = parser.parse_args()
    main(args.input_file)

