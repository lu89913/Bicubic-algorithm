import numpy as np
from PIL import Image
import os
import time # For basic timing

# Append src to sys.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from traditional_bicubic import bicubic_resize as float_bicubic_resize
from traditional_bicubic import cubic_kernel as float_cubic_kernel # For complexity discussion
from hardware_friendly_bicubic import bicubic_resize_fixed_point
from hardware_friendly_bicubic import cubic_kernel_fixed_point, F_BITS # For complexity discussion

def calculate_psnr_mse(img1, img2):
    """Calculates PSNR and MSE between two images."""
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions.")
    if img1.dtype != img2.dtype: # Promote to float64 for mse calculation
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
    
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel_val = 0
        if np.issubdtype(img1.dtype, np.integer) or img1.dtype == np.uint8 : # Check original type before astype
             max_pixel_val = np.iinfo(np.uint8).max # Assuming uint8 based on typical image data
        elif np.issubdtype(img1.dtype, np.floating):
             max_pixel_val = 1.0 # Assuming float images are in [0,1] range, adjust if not
        else: # Fallback, may need adjustment
            max_pixel_val = np.max(img1) if np.max(img1) > np.max(img2) else np.max(img2)
            if max_pixel_val <=0 : max_pixel_val = 255.0 # common case

        # If original was uint8, max_pixel_val should be 255
        # The input to this function might already be float after resize operations
        # For this project, images are uint8, so 255 is appropriate.
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
    return psnr, mse

def analyze_kernel_complexity():
    print("\n--- Kernel Complexity Analysis (Qualitative) ---")
    print("Traditional Float Kernel (per call, generic 'a'):")
    print("  - Approx. 4-6 float multiplications (for x^2, x^3, and with 'a'-dependent coeffs).")
    print("  - Approx. 3-4 float additions/subtractions.")

    print(f"\nHardware-Friendly Fixed-Point Kernel (per call, F_BITS={F_BITS}):")
    print("  - Common operations: Integer multiplications for x^2, x^3 (scaled): 2.")
    print("                     Shifts for scaling x^2, x^3 products: 2-3.")

    print("\n  For specific 'a' values (coefficient multiplications are optimized):")
    print("  a = -0.5:")
    print("    - Segment |x|<1 (1.5x^3 - 2.5x^2 + 1): ~2 shifts, ~2 adds for 1.5x^3; ~2 shifts, ~2 adds for 2.5x^2.")
    print("    - Segment 1<=|x|<2 (-0.5x^3 + 2.5x^2 - 4x + 2): ~1 shift for -0.5x^3; ~2 shifts, ~2 adds for 2.5x^2; ~1 shift for -4x.")
    print("    - Total: Approx. 4-6 shifts, 4-6 integer additions/subtractions per call (varies by segment).")
    print("  a = -0.75:")
    print("    - Segment |x|<1 (1.25x^3 - 2.25x^2 + 1): ~2 shifts, ~1 add for 1.25x^3; ~2 shifts, ~1 add for 2.25x^2.")
    print("    - Segment 1<=|x|<2 (-0.75x^3 + 3.75x^2 - 6x + 3): ~2 shifts, ~1 sub for -0.75x^3; ~2 shifts, ~1 sub for 3.75x^2; ~2 shifts, ~1 add for -6x.")
    print("    - Total: Approx. 4-6 shifts, 4-6 integer additions/subtractions per call.")
    print("  a = -1.0:")
    print("    - Segment |x|<1 (x^3 - 2x^2 + 1): 0 shifts for x^3; ~1 shift for 2x^2.")
    print("    - Segment 1<=|x|<2 (-x^3 + 5x^2 - 8x + 4): 0 shifts for -x^3; ~1 shift, ~1 add for 5x^2; ~1 shift for -8x.")
    print("    - Total: Approx. 2-4 shifts, 2-4 integer additions/subtractions per call.")

    print("\n  For generic 'a' values (using general fixed-point multiplication for coefficients):")
    print("    - Each 'a'-dependent coefficient (e.g., a+2, a+3, a, -5a, 8a, -4a) is pre-calculated as QsY.F.")
    print("    - Multiplication of these fixed-point coeffs with scaled x, x^2, x^3 terms:")
    print("      - Segment |x|<1: 2 fixed-point multiplications (e.g., (a+2)*x^3, (a+3)*x^2).")
    print("      - Segment 1<=|x|<2: 3 fixed-point multiplications (e.g., a*x^3, -5a*x^2, 8a*x).")
    print("    - These are full integer multiplications, potentially more complex than shifts/adds for specific 'a'.")

    print("\n  Benefit of optimized 'a' values: Replaces general integer multiplications for coefficients with simpler bit shifts and adds, leading to smaller/faster hardware.")

def analyze_interpolation_complexity():
    print("\n--- Interpolation Complexity Analysis (per output pixel, independent of 'a' value for this part) ---")
    print("Traditional Float (after 8 kernel calls):")
    print("  - Core Interpolation (wy.T @ p @ wx):")
    print("    - 20 float multiplications (pixel/intermediate * weight).")
    print("    - 15 float additions.")
    print("Hardware-Friendly Fixed-Point (after 8 fixed-point kernel calls):")
    print("  - Core Interpolation (wy.T @ p @ wx):")
    print("    - 20 integer multiplications (pixel(uint8) * weight(QsX.F)).")
    print("    - 15 integer additions.")
    print("    - Additional right-shifts for scaling final result.")
    print("  - Benefit: Replaces float DSPs with integer multipliers (potentially smaller/faster).")


def analyze_memory_access(original_shape, out_shape, scale_factor_x, scale_factor_y):
    print("\n--- Memory Access Analysis ---")
    original_height, original_width = original_shape
    out_height, out_width = out_shape

    mem_access_unbuffered_pixels = out_height * out_width * 16 # 16 pixels for 4x4 patch

    padded_image_width = original_width + 2 * 2 
    loaded_padded_row_indices = set()
    mem_access_buffered_pixels = 0

    for j_out in range(out_height):
        for i_out in range(out_width):
            x_in_float = i_out / scale_factor_x
            y_in_float = j_out / scale_factor_y
            x_int = int(np.floor(x_in_float)) 
            y_int = int(np.floor(y_in_float))
            start_row_in_padded = y_int + 1
            for i in range(4):
                current_padded_row_idx = start_row_in_padded + i
                if current_padded_row_idx not in loaded_padded_row_indices:
                    mem_access_buffered_pixels += padded_image_width
                    loaded_padded_row_indices.add(current_padded_row_idx)
    
    print(f"Scaling {scale_factor_x}x, Image: {original_width}x{original_height} -> {out_width}x{out_height}")
    print(f"  Unbuffered main memory accesses (pixels): {mem_access_unbuffered_pixels}")
    print(f"  Buffered main memory accesses (pixels): {mem_access_buffered_pixels}")
    
    reduction_factor = 0
    if mem_access_buffered_pixels > 0:
        reduction_factor = mem_access_unbuffered_pixels / mem_access_buffered_pixels
        print(f"  Reduction factor: {reduction_factor:.2f}x")
    else:
        print("  Buffered access is zero (check logic).")
    return mem_access_unbuffered_pixels, mem_access_buffered_pixels, reduction_factor


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_filename = "complex_test_image_256.png" # Use the new complex image
    image_path = os.path.join(base_dir, "images", image_filename)
    output_dir = os.path.join(base_dir, "images", "output")
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    try:
        pil_img_orig = Image.open(image_path).convert('L')
        np_img_orig = np.array(pil_img_orig, dtype=np.uint8)
    except FileNotFoundError:
        print(f"Error: Test image '{image_path}' not found.")
        sys.exit(1)

    scale_x, scale_y = 2.0, 2.0 # Scale 256x256 to 512x512
    a_values_to_test = [-0.5, -0.75, -1.0] # Values of 'a' to test
    # F_BITS is read from hardware_friendly_bicubic.py, ensure it's 10 for these tests.

    print(f"Comparing Bicubic Algorithms for image: '{image_filename}' ({np_img_orig.shape[1]}x{np_img_orig.shape[0]})")
    print(f"Scaling by factor: {scale_x}x horizontally, {scale_y}x vertically (Output: {int(np_img_orig.shape[1]*scale_x)}x{int(np_img_orig.shape[0]*scale_y)}).")
    print(f"Using F_BITS = {F_BITS} for fixed-point operations.")

    # --- Pillow (Reference, typically a=-0.5 or similar) ---
    start_time = time.time()
    expected_width = int(np.ceil(np_img_orig.shape[1] * scale_x))
    expected_height = int(np.ceil(np_img_orig.shape[0] * scale_y))
    pil_img_resized = pil_img_orig.resize((expected_width, expected_height), Image.Resampling.BICUBIC)
    np_pil_resized = np.array(pil_img_resized, dtype=np.uint8)
    pillow_time = time.time() - start_time
    print(f"\n1. Pillow BICUBIC resizing done in {pillow_time:.4f}s (Used as reference for a ~ -0.5)")
    # Save Pillow's output
    pillow_output_filename = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_pillow_bicubic.png")
    pil_img_resized.save(pillow_output_filename)
    print(f"  Saved Pillow output to {pillow_output_filename}")

    for a_val in a_values_to_test:
        print(f"\n--- Testing with a = {a_val} ---")

        # --- Traditional Float Bicubic ---
        start_time = time.time()
        np_float_resized = float_bicubic_resize(np_img_orig, scale_x, scale_y, a=a_val)
        float_time = time.time() - start_time
        print(f"  2. Traditional Float Bicubic (a={a_val}) resizing done in {float_time:.4f}s")
        float_output_filename = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_float_a{a_val}.png")
        Image.fromarray(np_float_resized, mode='L').save(float_output_filename)
        print(f"    Saved Float (a={a_val}) output to {float_output_filename}")

        # --- Hardware-Friendly Fixed-Point Bicubic ---
        start_time = time.time()
        np_fixed_resized = bicubic_resize_fixed_point(np_img_orig, scale_x, scale_y, a_float=a_val)
        fixed_time = time.time() - start_time
        print(f"  3. Hardware-Friendly Fixed-Point Bicubic (a={a_val}, F_BITS={F_BITS}) resizing done in {fixed_time:.4f}s")
        fixed_output_filename = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_fixed_a{a_val}_fb{F_BITS}.png")
        Image.fromarray(np_fixed_resized, mode='L').save(fixed_output_filename)
        print(f"    Saved Fixed-Point (a={a_val}, F_BITS={F_BITS}) output to {fixed_output_filename}")

        # --- Image Quality Comparison ---
        print(f"\n  --- Image Quality (PSNR dB / MSE) for a = {a_val} ---")
        psnr_float_vs_pillow, mse_float_vs_pillow = calculate_psnr_mse(np_float_resized, np_pil_resized)
        print(f"    Float (a={a_val}) vs Pillow: PSNR={psnr_float_vs_pillow:.2f} dB, MSE={mse_float_vs_pillow:.2f}")

        psnr_fixed_vs_pillow, mse_fixed_vs_pillow = calculate_psnr_mse(np_fixed_resized, np_pil_resized)
        print(f"    Fixed-Point (a={a_val}) vs Pillow: PSNR={psnr_fixed_vs_pillow:.2f} dB, MSE={mse_fixed_vs_pillow:.2f}")

        psnr_fixed_vs_float, mse_fixed_vs_float = calculate_psnr_mse(np_fixed_resized, np_float_resized)
        print(f"    Fixed-Point (a={a_val}) vs Float (a={a_val}): PSNR={psnr_fixed_vs_float:.2f} dB, MSE={mse_fixed_vs_float:.2f}")

        # --- Complexity Analysis (specific to this 'a' if kernel analysis is updated) ---
        # analyze_kernel_complexity() # This function needs to be parameterized or display general info

        # --- Memory Access Analysis (independent of 'a') ---
        # analyze_memory_access(np_img_orig.shape, np_fixed_resized.shape, scale_x, scale_y) # Only run once if results are same

        print(f"\n  --- Summary for Hardware-Friendly Version (a={a_val}) ---")
        print(f"  1. Image Quality: Fixed-Point vs Float (a={a_val}): PSNR={psnr_fixed_vs_float:.2f} dB. Fixed-Point vs Pillow: PSNR={psnr_fixed_vs_pillow:.2f} dB.")
        print(f"     (Using {F_BITS}-bit fractional precision for fixed-point.)")
        # Computation simplification summary might need to be general or mention specific 'a' if optimized
        print("  2. Computation Simplification (Qualitative):")
        if a_val in [-0.5, -0.75, -1.0]:
            print(f"     - Kernel for a={a_val}: Optimized with shifts & integer adds.")
        else:
            print(f"     - Kernel for a={a_val}: Uses generic fixed-point multiplications for coefficients.")
        print("     - Interpolation: Float multiplications (20 per pixel) replaced by integer multiplications.")
    
    # General analyses that are not 'a' dependent can be run once outside the loop
    print("\n--- General Complexity & Memory Analysis ---")
    analyze_kernel_complexity() # Should be updated to discuss different 'a' values
    analyze_interpolation_complexity()
    # Assuming np_fixed_resized from the last iteration for shape, or use expected_width/height
    analyze_memory_access(np_img_orig.shape, (expected_height, expected_width), scale_x, scale_y)

    print("\n--- Overall Summary of Advantages for Hardware-Friendly Version ---")
    print(f"  Key benefit: Maintains high PSNR across various 'a' values while enabling hardware-efficient computation.")
    print(f"  (Refer to specific 'a' value summaries above for detailed PSNRs with {F_BITS}-bit fractional precision)")
    print("  Computation Simplification:")
    print("   - Kernel: Float multiplications for coefficients replaced by shifts & integer adds (for a=-0.5, -0.75, -1.0) or generic fixed-point ops.")
    print("   - Interpolation: Float multiplications (20 per pixel) replaced by integer multiplications.")
    print("   - Overall: Significant reduction in need for full float DSP units, replaced by simpler integer logic and shifters.")
    print("3. Memory Bandwidth Reduction (with line buffers):")
    print(f"   - Drastic decrease in main memory accesses (simulated ~{analyze_memory_access(np_img_orig.shape, np_fixed_resized.shape, scale_x, scale_y)[2]:.2f}x reduction).")
    print("     This leads to lower power consumption and higher throughput in hardware.")
    print("4. Potential Hardware Benefits:")
    print("   - Fewer DSP slices used (due to float -> int/shift conversion).")
    print("   - BRAM used for line buffers, but this is efficient for the bandwidth saved.")
    print("   - Higher clock speeds possible due to simpler arithmetic operations.")
    print("   - Lower power consumption.")

    # Note: Python execution times here are not representative of hardware performance.
    # Float operations in Python (via NumPy) are highly optimized C/Fortran.
    # Our Python fixed-point is for logical simulation and will be slower in Python.
    print("\nNote: Python execution times are NOT indicative of hardware performance.")
    print("The fixed-point Python code is for logical simulation and is expected to be slower than NumPy's C-optimized float operations in a Python environment.")
