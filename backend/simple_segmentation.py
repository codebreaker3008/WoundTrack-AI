"""
Simple Wound Segmentation Without Training
Uses traditional computer vision techniques
"""

import cv2
import numpy as np

class SimpleWoundSegmenter:
    """
    Simple wound segmentation using color-based and edge detection
    No training required!
    """
    
    def __init__(self):
        pass
    
    def segment_wound(self, image):
        """
        Segment wound from image using color and edge detection
        
        Args:
            image: RGB image (numpy array)
        
        Returns:
            Binary mask (numpy array)
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Method 1: Red/Pink color detection (common in wounds)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Method 2: Pink/flesh tones
        lower_pink = np.array([0, 10, 60])
        upper_pink = np.array([20, 150, 255])
        mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)
        
        # Method 3: Yellow/brown (infected wounds)
        lower_yellow = np.array([15, 30, 50])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Combine masks
        mask_color = cv2.bitwise_or(mask_red, mask_pink)
        mask_color = cv2.bitwise_or(mask_color, mask_yellow)
        
        # Method 4: Edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to create regions
        kernel = np.ones((5, 5), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Combine color and edge information
        mask_combined = cv2.bitwise_or(mask_color, edges_dilated)
        
        # Morphological operations to clean up
        kernel_close = np.ones((15, 15), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel_close)
        
        # Remove small noise
        kernel_open = np.ones((5, 5), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel_open)
        
        # Fill holes
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Get largest contour (assumed to be wound)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create clean mask with filled contour
            final_mask = np.zeros_like(mask_cleaned)
            cv2.drawContours(final_mask, [largest_contour], -1, 255, -1)
            
            return final_mask
        
        # Fallback: If no contours, use center region
        h, w = image.shape[:2]
        fallback_mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        axes = (int(w * 0.2), int(h * 0.2))
        cv2.ellipse(fallback_mask, center, axes, 0, 0, 360, 255, -1)
        
        return fallback_mask
    
    def segment_with_grabcut(self, image):
        """
        More sophisticated segmentation using GrabCut algorithm
        
        Args:
            image: RGB image (numpy array)
        
        Returns:
            Binary mask (numpy array)
        """
        h, w = image.shape[:2]
        
        # Initialize mask
        mask = np.zeros((h, w), np.uint8)
        
        # Define rectangle around center (assumed wound location)
        rect = (int(w * 0.2), int(h * 0.2), int(w * 0.6), int(h * 0.6))
        
        # Initialize GrabCut
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Convert to BGR for GrabCut
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            # Run GrabCut
            cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Create binary mask
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            result_mask = mask2 * 255
            
            # Clean up
            kernel = np.ones((7, 7), np.uint8)
            result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_CLOSE, kernel)
            result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_OPEN, kernel)
            
            return result_mask
        except:
            # Fallback to simple method
            return self.segment_wound(image)
    
    def segment_adaptive(self, image):
        """
        Adaptive segmentation that tries multiple methods
        
        Args:
            image: RGB image (numpy array)
        
        Returns:
            Binary mask (numpy array)
        """
        # Try color-based first
        mask_color = self.segment_wound(image)
        
        # Calculate quality of mask
        mask_area = np.sum(mask_color > 0)
        total_area = mask_color.shape[0] * mask_color.shape[1]
        area_ratio = mask_area / total_area
        
        # If mask seems reasonable, use it
        if 0.05 < area_ratio < 0.6:
            return mask_color
        
        # Otherwise try GrabCut
        print("Color-based segmentation failed, trying GrabCut...")
        mask_grabcut = self.segment_with_grabcut(image)
        
        mask_area = np.sum(mask_grabcut > 0)
        area_ratio = mask_area / total_area
        
        if 0.05 < area_ratio < 0.6:
            return mask_grabcut
        
        # Final fallback: center ellipse
        print("All methods failed, using center region...")
        h, w = image.shape[:2]
        fallback_mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        axes = (int(w * 0.15), int(h * 0.12))
        cv2.ellipse(fallback_mask, center, axes, 0, 0, 360, 255, -1)
        
        return fallback_mask


def test_segmentation():
    """Test the segmentation on a sample image"""
    import matplotlib.pyplot as plt
    
    # Create a test image with a red circular "wound"
    test_image = np.ones((512, 512, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Add a red circular wound
    center = (256, 256)
    radius = 80
    cv2.circle(test_image, center, radius, (180, 50, 50), -1)  # Red wound
    
    # Add some texture
    noise = np.random.randint(-20, 20, test_image.shape, dtype=np.int16)
    test_image = np.clip(test_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Test segmentation
    segmenter = SimpleWoundSegmenter()
    
    mask1 = segmenter.segment_wound(test_image)
    mask2 = segmenter.segment_adaptive(test_image)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(test_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask1, cmap='gray')
    axes[1].set_title('Color-based Segmentation')
    axes[1].axis('off')
    
    axes[2].imshow(mask2, cmap='gray')
    axes[2].set_title('Adaptive Segmentation')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('segmentation_test.png', dpi=150, bbox_inches='tight')
    print("✅ Test complete! Check 'segmentation_test.png'")
    plt.show()


if __name__ == "__main__":
    print("Testing Simple Wound Segmentation...")
    test_segmentation()