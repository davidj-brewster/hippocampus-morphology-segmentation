import logging
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, filters, measure
import cv2
from PIL import Image

def largest_connected_component(binary_mask):
    labeled_mask, num_labels = measure.label(binary_mask, connectivity=1, return_num=True)
    if num_labels == 0:
        return binary_mask
    largest_label = 1 + np.argmax(np.bincount(labeled_mask.flat)[1:])
    return labeled_mask == largest_label

def skull_stripping(image, output_path='/tmp/', min_size=1000, area_threshold=6000):
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

def region_of_interest(brain_image, edges, region):
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

def extract_region(brain_image, edges, region, min_area=500, max_area=5000):
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


def detect_hippocampus_anomalies(left_mask, right_mask, threshold=0.3):
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

    # Perform skull stripping
    skull_stripped_image = skull_stripping(image, output_path_base)

    # Perform edge detection
    edges = edge_detection(skull_stripped_image, output_path_base)

    # Extract hippocampal regions
    left_mask = extract_region(skull_stripped_image, edges, 'left_hippocampus')
    right_mask = extract_region(skull_stripped_image, edges, 'right_hippocampus')

    # Detect hippocampal anomalies
    anomalies = detect_hippocampus_anomalies(left_mask, right_mask)

    # Visualize results
    visualize_results(image, ['left_hippocampus', 'right_hippocampus'], 
                      {'left_hippocampus': left_mask, 'right_hippocampus': right_mask}, anomalies)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Brain MRI analysis")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input image file')
    
    args = parser.parse_args()
    main(args.input_file)

