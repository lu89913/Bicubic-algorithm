import unittest
import numpy as np
from PIL import Image
import os # For path joining

# Append src to sys.path to allow direct import of modules from src
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from traditional_bicubic import bicubic_resize as float_bicubic_resize
from hardware_friendly_bicubic import bicubic_resize_fixed_point, F_BITS

class TestHardwareFriendlyBicubic(unittest.TestCase):

    def setUp(self):
        # Construct path to image relative to this test file's directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        image_path_relative = os.path.join('..', 'images', 'gradient.png')
        self.image_path = os.path.normpath(os.path.join(base_dir, image_path_relative))

        try:
            self.original_pil_img = Image.open(self.image_path).convert('L') # Grayscale
            self.original_np_img = np.array(self.original_pil_img, dtype=np.uint8)
        except FileNotFoundError:
            self.fail(f"Test image not found at {self.image_path}. Ensure images/gradient.png exists.")

        self.scale_factor_x = 1.5
        self.scale_factor_y = 1.5
        self.kernel_a_param = -0.5 # Default 'a' for baseline tests

    def test_fixed_point_resize_against_float(self):
        """
        Test the fixed-point bicubic resize implementation against the float version.
        Evaluates the precision of the fixed-point arithmetic.
        """
        float_resized_np = float_bicubic_resize(self.original_np_img,
                                                self.scale_factor_x,
                                                self.scale_factor_y,
                                                a=self.kernel_a_param)

        fixed_resized_np = bicubic_resize_fixed_point(self.original_np_img,
                                                      self.scale_factor_x,
                                                      self.scale_factor_y,
                                                      a_float=self.kernel_a_param)

        self.assertEqual(fixed_resized_np.shape, float_resized_np.shape,
                         "Shape mismatch between fixed-point and float output.")

        # Calculate Mean Squared Error (MSE)
        mse = np.mean((fixed_resized_np.astype(np.float64) - float_resized_np.astype(np.float64))**2)

        # Calculate PSNR
        if mse == 0:
            psnr = float('inf')
        else:
            max_pixel_val = 255.0 # For uint8 images
            psnr = 20 * np.log10(max_pixel_val / np.sqrt(mse))

        print(f"\nFixed-Point (F_BITS={F_BITS}) vs Float Bicubic:")
        print(f"  MSE: {mse:.4f}")
        print(f"  PSNR: {psnr:.4f} dB")

        # Set a PSNR threshold. For F_BITS=8, we expect good quality.
        # A common threshold for "good" similarity is > 30 dB.
        # For "very good" or "visually indistinguishable" might be > 40 dB.
        # Let's aim for > 35 dB for F_BITS=8 initially.
        psnr_threshold = 35.0
        self.assertGreaterEqual(psnr, psnr_threshold,
                             f"PSNR ({psnr:.4f} dB) is below threshold ({psnr_threshold} dB). "
                             f"Fixed-point precision with F_BITS={F_BITS} may be too low or there's an issue.")

    def test_output_shape_fixed_point(self):
        """Test if the output image from fixed-point has the correct shape."""
        resized_img = bicubic_resize_fixed_point(self.original_np_img,
                                                 self.scale_factor_x,
                                                 self.scale_factor_y,
                                                 a_float=self.kernel_a_param)

        original_height, original_width = self.original_np_img.shape
        expected_height = int(np.ceil(original_height * self.scale_factor_y))
        expected_width = int(np.ceil(original_width * self.scale_factor_x))

        self.assertEqual(resized_img.shape, (expected_height, expected_width))

    def test_memory_access_simulation(self):
        """Simulates and compares memory access patterns with and without line buffering."""
        original_height, original_width = self.original_np_img.shape
        out_height = int(np.ceil(original_height * self.scale_factor_y))
        out_width = int(np.ceil(original_width * self.scale_factor_x))

        # Unbuffered access:
        # Each output pixel requires a 4x4 patch.
        # This is a simplified view where each of the 16 pixels is an independent read from main memory.
        mem_access_unbuffered_pixels = out_height * out_width * 16

        # Buffered access simulation:
        # Assumes a line buffer that holds the necessary rows from the (padded) input image.
        # Cost is incurred when a required row is not yet in the buffer and needs to be loaded.

        # Determine the width of the padded image, as this is what a line buffer would store.
        # pad_width=2 for bicubic (kernel support of 2 pixels on each side of center point)
        padded_image_width = original_width + 2 * 2

        loaded_padded_row_indices = set() # Stores indices of rows from the conceptual padded input
        mem_access_buffered_pixels = 0

        for j_out in range(out_height):
            for i_out in range(out_width):
                # Map output pixel to input image coordinates
                x_in_float = i_out / self.scale_factor_x
                y_in_float = j_out / self.scale_factor_y

                # Determine the top-left integer coordinate of the 4x4 grid in the *original* image
                x_int = int(np.floor(x_in_float))
                y_int = int(np.floor(y_in_float))

                # The 4x4 patch is taken from input rows y_int-1, y_int, y_int+1, y_int+2 (original indexing)
                # These correspond to padded image rows:
                # original y_int-1 -> padded y_int+2-1 = y_int+1
                # original y_int   -> padded y_int+2
                # original y_int+1 -> padded y_int+2+1 = y_int+3
                # original y_int+2 -> padded y_int+2+2 = y_int+4
                # So, the rows in the padded image are from y_int+1 to y_int+4.

                # Indices of the 4 rows needed from the *padded* image for the current 4x4 patch
                # The patch starts at padded_image[y_int+1, x_int+1]
                start_row_in_padded = y_int + 1 # Top row of the 4x4 patch in padded coordinates

                for i in range(4): # Iterate over the 4 rows of the 4x4 patch
                    current_padded_row_idx = start_row_in_padded + i
                    if current_padded_row_idx not in loaded_padded_row_indices:
                        # Simulate loading this row from main memory into the line buffer
                        mem_access_buffered_pixels += padded_image_width
                        loaded_padded_row_indices.add(current_padded_row_idx)

        print(f"\nMemory Access Simulation (Scaling {self.scale_factor_x}x, Image: {original_width}x{original_height} -> {out_width}x{out_height}):")
        print(f"  Unbuffered main memory accesses (pixels): {mem_access_unbuffered_pixels}")
        print(f"  Buffered main memory accesses (pixels): {mem_access_buffered_pixels}")

        reduction_factor = 0
        if mem_access_buffered_pixels > 0 :
            reduction_factor = mem_access_unbuffered_pixels / mem_access_buffered_pixels
            print(f"  Reduction factor: {reduction_factor:.2f}x")
        else:
            print("  Buffered access is zero (check logic if unexpected).")

        # The number of unique rows loaded should generally be small, close to in_height + 3
        # print(f"  Number of unique rows loaded into buffer: {len(loaded_padded_row_indices)}")

        # Expect a significant reduction
        self.assertGreater(reduction_factor, 3.0, "Line buffer should provide at least ~4x reduction for 4x4 kernel.")
        # For a 4-line buffer and 4x4 kernel, expect reuse of 3 out of 4 lines typically.
        # Ideal steady state is 1 new row per 4 used, so factor of 4. Start/end effects reduce this.

if __name__ == '__main__':
    unittest.main()
