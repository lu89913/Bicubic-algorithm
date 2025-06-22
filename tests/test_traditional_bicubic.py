import unittest
import numpy as np
from PIL import Image
from src.traditional_bicubic import bicubic_resize

class TestTraditionalBicubic(unittest.TestCase):

    def setUp(self):
        # Load the test image
        try:
            self.image_path = "images/gradient.png"
            self.original_pil_img = Image.open(self.image_path).convert('L') # Grayscale
            self.original_np_img = np.array(self.original_pil_img, dtype=np.uint8)
        except FileNotFoundError:
            # Fallback if run from a different CWD, try one level up for images dir
            try:
                self.image_path = "../images/gradient.png"
                self.original_pil_img = Image.open(self.image_path).convert('L')
                self.original_np_img = np.array(self.original_pil_img, dtype=np.uint8)
            except FileNotFoundError:
                self.fail("Test image not found. Make sure images/gradient.png exists.")

        self.scale_factor_x = 1.5
        self.scale_factor_y = 1.5
        # Pillow uses a=-0.5 for its bicubic kernel by default in recent versions,
        # or something very close to it.
        self.kernel_a_param = -0.5 

    def test_bicubic_resize_output_shape(self):
        """Test if the output image has the correct shape."""
        resized_img = bicubic_resize(self.original_np_img, self.scale_factor_x, self.scale_factor_y, a=self.kernel_a_param)
        
        expected_height = int(np.ceil(self.original_np_img.shape[0] * self.scale_factor_y))
        expected_width = int(np.ceil(self.original_np_img.shape[1] * self.scale_factor_x))
        
        self.assertEqual(resized_img.shape, (expected_height, expected_width))

    def test_bicubic_resize_against_pillow(self):
        """
        Test the custom bicubic resize implementation against Pillow's BICUBIC filter.
        A small difference is expected due to potential variations in implementation details
        (e.g., boundary handling, exact arithmetic precision, kernel definition details).
        """
        custom_resized_np = bicubic_resize(self.original_np_img, 
                                           self.scale_factor_x, 
                                           self.scale_factor_y, 
                                           a=self.kernel_a_param)

        expected_height = custom_resized_np.shape[0]
        expected_width = custom_resized_np.shape[1]

        # Pillow's resize
        # Image.BICUBIC is the standard one.
        # Note: Pillow's internal implementation details might vary slightly.
        pillow_resized_pil = self.original_pil_img.resize((expected_width, expected_height), Image.Resampling.BICUBIC)
        pillow_resized_np = np.array(pillow_resized_pil, dtype=np.uint8)

        self.assertEqual(custom_resized_np.shape, pillow_resized_np.shape, "Shape mismatch between custom and Pillow output.")

        # Calculate Mean Squared Error (MSE)
        mse = np.mean((custom_resized_np.astype(np.float64) - pillow_resized_np.astype(np.float64))**2)
        
        # Set a threshold for MSE. This might need adjustment.
        # Differences can arise from:
        # 1. Boundary conditions: We used 'reflect'. Pillow's specific boundary handling for .resize might differ.
        # 2. Exact floating point arithmetic order.
        # 3. Pillow might use a slightly different formulation or internal rounding for coordinates.
        # For a 0-255 range, an MSE of, say, 1.0 means an average pixel difference of 1.
        # An MSE of up to 5-10 might be acceptable for "visually similar" with slight differences.
        # Let's start with a threshold of 10.0 and see.
        mse_threshold = 10.0 
        print(f"\nMSE between custom Bicubic and Pillow's Bicubic: {mse:.4f}")

        # Calculate PSNR
        if mse == 0:
            psnr = float('inf') # Or a very high number like 100 dB
        else:
            max_pixel_val = 255.0
            psnr = 20 * np.log10(max_pixel_val / np.sqrt(mse))
        print(f"PSNR between custom Bicubic and Pillow's Bicubic: {psnr:.4f} dB")

        self.assertLessEqual(mse, mse_threshold, 
                             f"MSE ({mse:.4f}) exceeds threshold ({mse_threshold}). "
                             "The implementation might differ significantly from Pillow's BICUBIC or have an issue.")

if __name__ == '__main__':
    unittest.main()
