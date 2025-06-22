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
    print("Traditional Float Kernel (per call, a=-0.5):")
    print("  - Approx. 4-6 float multiplications (for x^2, x^3, and with coeffs like 1.5, 2.5).")
    print("  - Approx. 2-3 float additions/subtractions.")
    print("Hardware-Friendly Fixed-Point Kernel (per call, a=-0.5, F_BITS={F_BITS}):")
    print("  - Integer multiplications for x^2, x^3: 2 (e.g., Q2.F * Q2.F -> Q4.2F).")
    print("  - Coefficient 'multiplications' (1.5, 2.5, -0.5, -4.0) replaced by:")
    print("    - Bit shifts (left/right): Approx. 3-4.")
    print("    - Integer additions/subtractions: Approx. 3-4.")
    print("  - Benefit: Eliminates float DSPs for coefficients, uses basic logic ops.")

def analyze_interpolation_complexity():
    print("\n--- Interpolation Complexity Analysis (per output pixel) ---")
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
    image_filename = "gradient.png" # Make sure this is in 'images' directory
    image_path = os.path.join(base_dir, "images", image_filename)

    try:
        pil_img_orig = Image.open(image_path).convert('L')
        np_img_orig = np.array(pil_img_orig, dtype=np.uint8)
    except FileNotFoundError:
        print(f"Error: Test image '{image_path}' not found.")
        sys.exit(1)

    scale_x, scale_y = 1.5, 1.5
    
    print(f"Comparing Bicubic Algorithms for image: '{image_filename}' ({np_img_orig.shape[1]}x{np_img_orig.shape[0]})")
    print(f"Scaling by factor: {scale_x}x horizontally, {scale_y}x vertically.")

    # --- Pillow (Reference) ---
    start_time = time.time()
    expected_width = int(np.ceil(np_img_orig.shape[1] * scale_x))
    expected_height = int(np.ceil(np_img_orig.shape[0] * scale_y))
    pil_img_resized = pil_img_orig.resize((expected_width, expected_height), Image.Resampling.BICUBIC)
    np_pil_resized = np.array(pil_img_resized, dtype=np.uint8)
    pillow_time = time.time() - start_time
    print(f"\n1. Pillow BICUBIC resizing done in {pillow_time:.4f}s")

    # --- Traditional Float Bicubic ---
    start_time = time.time()
    np_float_resized = float_bicubic_resize(np_img_orig, scale_x, scale_y)
    float_time = time.time() - start_time
    print(f"2. Traditional Float Bicubic resizing done in {float_time:.4f}s")

    # --- Hardware-Friendly Fixed-Point Bicubic ---
    start_time = time.time()
    np_fixed_resized = bicubic_resize_fixed_point(np_img_orig, scale_x, scale_y)
    fixed_time = time.time() - start_time
    print(f"3. Hardware-Friendly Fixed-Point Bicubic resizing done in {fixed_time:.4f}s")

    # --- Image Quality Comparison ---
    print("\n--- Image Quality (PSNR dB / MSE) ---")
    psnr_float_vs_pillow, mse_float_vs_pillow = calculate_psnr_mse(np_float_resized, np_pil_resized)
    print(f"  Float vs Pillow: PSNR={psnr_float_vs_pillow:.2f} dB, MSE={mse_float_vs_pillow:.2f}")

    psnr_fixed_vs_pillow, mse_fixed_vs_pillow = calculate_psnr_mse(np_fixed_resized, np_pil_resized)
    print(f"  Fixed-Point vs Pillow: PSNR={psnr_fixed_vs_pillow:.2f} dB, MSE={mse_fixed_vs_pillow:.2f}")
    
    psnr_fixed_vs_float, mse_fixed_vs_float = calculate_psnr_mse(np_fixed_resized, np_float_resized)
    print(f"  Fixed-Point vs Float: PSNR={psnr_fixed_vs_float:.2f} dB, MSE={mse_fixed_vs_float:.2f}")

    # --- Complexity Analysis ---
    analyze_kernel_complexity()
    analyze_interpolation_complexity()
    
    # --- Memory Access Analysis ---
    analyze_memory_access(np_img_orig.shape, np_fixed_resized.shape, scale_x, scale_y)

    print("\n--- Summary of Advantages for Hardware-Friendly Version ---")
    print(f"1. Image Quality: Maintains high PSNR ({psnr_fixed_vs_float:.2f} dB against float version, "
          f"{psnr_fixed_vs_pillow:.2f} dB against Pillow) with {F_BITS}-bit fractional precision.")
    print("2. Computation Simplification:")
    print("   - Kernel: Float multiplications for coefficients replaced by shifts & integer adds.")
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
