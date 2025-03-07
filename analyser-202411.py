"""
HippocampusExtractor - A class for extracting and analyzing hippocampal regions from brain images.

This class provides a comprehensive pipeline for processing brain images, detecting edges,
extracting the left and right hippocampal regions, and verifying the extraction results.
The class supports both Canny and Gabor-based edge detection, and includes methods for
refining the extracted hippocampal masks and verifying their anatomical correctness.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology
from skimage.feature import canny
from skimage.filters import gabor_kernel, gaussian
from skimage.metrics import structural_similarity as ssim
from medpy.filter.binary import largest_connected_component
from scipy.ndimage.measurements import center_of_mass

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HippocampusExtractor:
    """
    A class for extracting and analyzing hippocampal regions from brain images.

    Attributes:
        image_path (str): Path to the input image.
        mri_path (str): Path to the MRI image (optional).
        ct_path (str): Path to the CT image (optional).
        pet_path (str): Path to the PET image (optional).
        image (np.ndarray): The input image.
        preprocessed_image (np.ndarray): The preprocessed image.
        brain_image (np.ndarray): The brain image after skull stripping.
        edges (np.ndarray): The edge map of the brain image.
        left_mask (np.ndarray): The mask for the left hippocampal region.
        right_mask (np.ndarray): The mask for the right hippocampal region.
    """

    def __init__(self, image_path, mri_path=None, ct_path=None, pet_path=None):
        self.image_path = image_path
        self.mri_path = mri_path
        self.ct_path = ct_path
        self.pet_path = pet_path
        self.image = None
        self.preprocessed_image = None
        self.brain_image = None
        self.edges = None
        self.left_mask = None
        self.right_mask = None

    def load_and_preprocess(self):
        """Load the image and preprocess it."""
        self.image = self._load_image(self.image_path)
        self.preprocessed_image = self._preprocess_image(self.image)

    def segment_brain(self):
        """Perform skull stripping to extract the brain region."""
        self.brain_image = self._skull_stripping(self.preprocessed_image)

    def detect_edges(self, sigma=1, low_threshold=0.1, high_threshold=0.5, use_gabor=False):
        """
        Detect edges in the brain image using the Canny algorithm or Gabor filters.

        Args:
            sigma (float): Standard deviation of the Gaussian kernel for Canny or Gabor.
            low_threshold (float): Low threshold for hysteresis in Canny edge detection.
            high_threshold (float): High threshold for hysteresis in Canny edge detection.
            use_gabor (bool): If True, use Gabor filters for edge detection; otherwise, use Canny.
        """
        if use_gabor:
            self.edges = self._gabor_edge_detection(self.brain_image, sigma)
        else:
            self.edges = self._canny_edge_detection(self.brain_image, sigma, low_threshold, high_threshold)

    def compare_edge_detection_methods(self):
        """Compare the results of Canny and Gabor edge detection methods."""
        try:
            logging.info("Comparing Canny and Gabor edge detection results")

            # Detect edges using Canny
            canny_edges = self._canny_edge_detection(self.brain_image, sigma=1, low_threshold=0.1, high_threshold=0.5)

            # Detect edges using Gabor
            gabor_edges = self._gabor_edge_detection(self.brain_image, sigma=1)

            # Calculate the structural similarity index (SSIM) between the two edge maps
            ssim_score = ssim(canny_edges, gabor_edges)
            logging.info(f"SSIM between Canny and Gabor edge maps: {ssim_score:.2f}")

            # Check if the edge maps are sufficiently similar
            if ssim_score < 0.8:
                logging.warning("Canny and Gabor edge detection results are significantly different. Consider adjusting parameters or investigating further.")
            else:
                logging.info("Canny and Gabor edge detection results are sufficiently similar. Proceeding with hippocampal segmentation.")

            return canny_edges, gabor_edges
        except Exception as e:
            logging.error(f"Error during edge detection comparison: {e}")
            raise

    def extract_hippocampi(self):
        """Extract the left and right hippocampal regions."""
        self.left_mask = self._extract_hippocampal_region(self.brain_image, self.edges, 'left')
        self.right_mask = self._extract_hippocampal_region(self.brain_image, self.edges, 'right')

        # Refine the hippocampal masks
        self.left_mask = self._refine_hippocampal_mask(self.brain_image, self.left_mask)
        self.right_mask = self._refine_hippocampal_mask(self.brain_image, self.right_mask)

        # Verify the hippocampal extraction
        self.verify_hippocampal_extraction()

    def _extract_hippocampal_region(self, brain_image, edges, side):
        """
        Extract the hippocampal region based on the side (left or right).

        Args:
            brain_image (np.ndarray): The preprocessed brain image.
            edges (np.ndarray): The edge map of the brain image.
            side (str): The side of the hippocampus to extract ('left' or 'right').

        Returns:
            np.ndarray: The mask for the extracted hippocampal region.
        """
        try:
            logging.info(f"Extracting {side} hippocampal region")
            height, width = brain_image.shape

            # Define the ROI using anatomical landmarks
            if side == 'left':
                roi_x1 = width // 6
                roi_x2 = width // 2
                roi_y1 = height // 4
                roi_y2 = 3 * height // 4
            elif side == 'right':
                roi_x1 = width // 2
                roi_x2 = 5 * width // 6
                roi_y1 = height // 4
                roi_y2 = 3 * height // 4
            else:
                raise ValueError("side must be 'left' or 'right'")

            roi = edges[roi_y1:roi_y2, roi_x1:roi_x2]

            # Label connected components and filter by size
            labeled_roi = measure.label(roi)
            regions = measure.regionprops(labeled_roi)
            filtered_regions = [r for r in regions if 200 < r.area < 80000]

            # Create the hippocampal mask
            hippocampal_mask = np.zeros((height, width // 2), dtype=np.bool_)
            if filtered_regions:
                largest_region = max(filtered_regions, key=lambda r: r.area)
                if side == 'left':
                    hippocampal_mask[roi_y1:roi_y2, :roi_x2 - roi_x1] = (labeled_roi == largest_region.label)
                elif side == 'right':
                    hippocampal_mask[roi_y1:roi_y2, roi_x1 - width // 2:] = (labeled_roi == largest_region.label)

            logging.info(f"{side.capitalize()} hippocampal region extracted with area: {np.sum(hippocampal_mask)}")
            return hippocampal_mask
        except Exception as e:
            logging.error(f"Error extracting hippocampal region: {e}")
            raise

    def _refine_hippocampal_mask(self, brain_image, mask):
        """
        Refine the hippocampal mask using morphological operations and intensity-based filtering.

        Args:
            brain_image (np.ndarray): The preprocessed brain image.
            mask (np.ndarray): The initial hippocampal mask.

        Returns:
            np.ndarray: The refined hippocampal mask.
        """
        try:
            logging.info("Refining hippocampal mask")

            # Apply morphological operations to smooth the mask
            mask = morphology.opening(mask, morphology.disk(3))
            mask = morphology.closing(mask, morphology.disk(3))

            # Apply intensity-based refinement
            mean_intensity = np.mean(brain_image[mask])
            mask = (brain_image > mean_intensity * 0.8) & mask

            return mask
        except Exception as e:
            logging.error(f"Error refining hippocampal mask: {e}")
            raise

    def verify_hippocampal_extraction(self):
        """
        Verify the extracted hippocampal regions.

        This method checks the following:
        1. Overlap between the left and right hippocampal masks (Dice coefficient).
        2. Anatomical position of the left and right hippocampal regions.
        3. Visibility of the hippocampus in the provided image slice.
        """
        try:
            logging.info("Verifying hippocampal region extraction")

            # Check if hippocampal masks are available
            if self.left_mask is None or self.right_mask is None:
                logging.warning("Hippocampal masks not available. Unable to verify extraction.")
                return

            # Check if the hippocampus is even visible in the image
            if np.sum(self.left_mask) < 100 and np.sum(self.right_mask) < 100:
                logging.warning("Hippocampus may not be visible in the provided image slice. Extraction results may not be reliable.")
                return

            # Compute the overlap between the left and right hippocampal masks
            overlap = np.logical_and(self.left_mask, self.right_mask)
            overlap_area = np.sum(overlap)
            total_area = np.sum(self.left_mask) + np.sum(self.right_mask)

            # Calculate the Dice coefficient as a measure of overlap
            dice_coefficient = 2 * overlap_area / total_area

            logging.info(f"Dice coefficient between left and right hippocampus: {dice_coefficient:.2f}")

            # Check if the Dice coefficient is within an acceptable range
            if dice_coefficient < 0.6:
                logging.warning("Hippocampal extraction results are not satisfactory. Consider adjusting parameters or investigating further.")

            # Check if the hippocampal regions are in the expected anatomical locations
            left_center = center_of_mass(self.left_mask)
            right_center = center_of_mass(self.right_mask)

            if left_center[0] > right_center[0]:
                logging.warning("Hippocampal regions are likely swapped. The left hippocampus should be on the left side of the image.")
            elif abs(left_center[0] - right_center[0]) < self.image.shape[1] / 6:
                logging.warning("Hippocampal regions are too close together. They may be inaccurately extracted.")

            logging.info("Hippocampal extraction verified successfully.")

        except Exception as e:
            logging.error(f"Error during hippocampal extraction verification: {e}")
            raise

    def compare_hippocampi(self):
        """
        Compare the areas of the left and right hippocampi.

        Returns:
            tuple: The areas of the left and right hippocampal regions.
        """
        left_area, right_area = self._compare_hippocampal_regions(self.left_mask, self.right_mask)
        return left_area, right_area

    def _compare_hippocampal_regions(self, left_mask, right_mask):
        """
        Compare the areas of the left and right hippocampal regions.

        Args:
            left_mask (np.ndarray): The mask for the left hippocampal region.
            right_mask (np.ndarray): The mask for the right hippocampal region.

        Returns:
            tuple: The areas of the left and right hippocampal regions.
        """
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

    def visualize_results(self):
        """Visualize the extracted hippocampal regions."""
        self._visualize_results(self.brain_image, self.edges[:, :self.edges.shape[1] // 2], self.edges[:, self.edges.shape[1] // 2:], self.left_mask, self.right_mask)

    def _load_image(self, file_path):
        """
        Load the image from the specified file path.

        Args:
            file_path (str): The path to the image file.

        Returns:
            np.ndarray: The loaded image.
        """
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

    def _preprocess_image(self, image):
        """
        Preprocess the image by normalizing the pixel values.

        Args:
            image (np.ndarray): The input image.

        Returns:
            np.ndarray: The preprocessed image.
        """
        try:
            logging.info("Starting image preprocessing")
            preprocessed_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            logging.info("Image normalization completed")
            return preprocessed_image
        except Exception as e:
            logging.error(f"Error during image preprocessing: {e}")
            raise

    def _skull_stripping(self, image):
        """
        Perform skull stripping to extract the brain region.

        Args:
            image (np.ndarray): The preprocessed image.

        Returns:
            np.ndarray: The brain image after skull stripping.
        """
        try:
            logging.info("Starting skull stripping")
            _, binary_mask = cv2.threshold((image * 255).astype(np.uint8), 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
            binary_mask = morphology.remove_small_objects(binary_mask > 0, min_size=5)
            binary_mask = morphology.remove_small_holes(binary_mask, area_threshold=5)
            brain_mask = largest_connected_component(binary_mask)
            brain_extracted = image * brain_mask
            logging.info("Skull stripping completed")
            return brain_extracted
        except Exception as e:
            logging.error(f"Error during skull stripping: {e}")
            raise

def _canny_edge_detection(self, image, sigma, low_threshold, high_threshold):
        """
        Detect edges in the image using the Canny algorithm.

        Args:
            image (np.ndarray): The input image.
            sigma (float): Standard deviation of the Gaussian kernel.
            low_threshold (float): Low threshold for hysteresis.
            high_threshold (float): High threshold for hysteresis.

        Returns:
            np.ndarray: The binary edge map.
        """
        try:
            logging.info("Starting Canny edge detection")
            edges = canny(image, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
            logging.info(f"Edge coordinates detected: {np.argwhere(edges).shape[0]} edges")
            return edges
        except Exception as e:
            logging.error(f"Error during Canny edge detection: {e}")
            raise

def _gabor_edge_detection(self, image, sigma):
    """
    Detect edges in the image using Gabor filters.

    Args:
        image (np.ndarray): The input image.
        sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
        np.ndarray: The binary edge map.
    """
    try:
        logging.info("Starting Gabor edge detection")
        gray_image = (image * 255).astype(np.uint8)
        gabor_edges = np.zeros_like(gray_image, dtype=np.float32)

        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            kernel = gabor_kernel(frequency=0.5, theta=theta, sigma_x=sigma, sigma_y=sigma)
            filtered_image = gaussian(gray_image, sigma=sigma) * np.real(cv2.filter2D(gray_image, cv2.CV_32F, kernel))
            gabor_edges = np.maximum(gabor_edges, filtered_image)
        logging.info(f"Edge coordinates detected: {np.argwhere(gabor_edges > 0).shape[0]} edges")
        return gabor_edges > 0.2
    except Exception as e:
        logging.error(f"Error during Gabor edge detection: {e}")
        raise
