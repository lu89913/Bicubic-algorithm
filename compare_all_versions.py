import numpy as np
import time
from skimage.metrics import peak_signal_noise_ratio

# Import resize functions from the different implementation files
try:
    from traditional_bicubic import imresize as imresize_traditional
except ImportError:
    print("Could not import imresize_traditional from traditional_bicubic.py")
    imresize_traditional = None

try:
    from optimized_bicubic_float import imresize_optimized_float
except ImportError:
    print("Could not import imresize_optimized_float from optimized_bicubic_float.py")
    imresize_optimized_float = None

try:
    from optimized_bicubic_fixed_point import imresize_fixed_point
except ImportError:
    print("Could not import imresize_fixed_point from optimized_bicubic_fixed_point.py")
    imresize_fixed_point = None


def create_test_image(height, width, channels=1, seed=42):
    """Generates a test image with a varied pattern."""
    np.random.seed(seed) # For reproducibility
    if channels == 1:
        # Create a more complex pattern than simple gradients
        img = np.zeros((height, width), dtype=np.float32)
        for _ in range(5): # Add a few random ellipses/circles
            x0, y0 = np.random.randint(0, width), np.random.randint(0, height)
            rx, ry = np.random.randint(width//10, width//3), np.random.randint(height//10, height//3)
            val = np.random.rand() * 128 + 64 # Random intensity
            Y, X = np.ogrid[:height, :width]
            mask = ((X - x0)/rx)**2 + ((Y - y0)/ry)**2 <= 1
            img[mask] += val
        img = np.clip(img, 0, 255)
        # Add some noise
        # noise = np.random.normal(0, 10, (height,width)).astype(np.float32)
        # img += noise
        # img = np.clip(img,0,255)
        return img.astype(np.uint8)
    elif channels == 3:
        img_r = create_test_image(height, width, channels=1, seed=seed)
        img_g = create_test_image(height, width, channels=1, seed=seed + 1)
        img_b = create_test_image(height, width, channels=1, seed=seed + 2)
        return np.stack((img_r, img_g, img_b), axis=2).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported number of channels: {channels}")

def calculate_psnr(img_true, img_test, data_range=255):
    """Calculates PSNR, handling potential all-zero diff warning."""
    if img_true.shape != img_test.shape:
        print(f"PSNR calculation error: Shapes mismatch! True: {img_true.shape}, Test: {img_test.shape}")
        return -1 # Or raise error

    # Ensure images are of the same type for PSNR calculation if one is float (e.g. after processing)
    # However, skimage handles this. Forcing uint8 if that's the common ground.
    # If inputs are already uint8, data_range=255 is fine.
    # If inputs are float, skimage psnr expects them to be in range [0,1] or specify data_range.
    # Since our true_image (traditional) is uint8, we use data_range=255.

    # Handle cases where images might be perfectly identical (PSNR = inf)
    if np.array_equal(img_true, img_test):
        return float('inf')

    try:
        psnr = peak_signal_noise_ratio(img_true, img_test, data_range=data_range)
    except ZeroDivisionError: # Should be caught by skimage, but as a fallback
        psnr = float('inf')
    return psnr


if __name__ == "__main__":
    input_height, input_width = 256, 256
    output_height, output_width = 512, 512
    target_output_shape = (output_height, output_width)

    print(f"Generating test images ({input_height}x{input_width})...")
    gray_image_orig = create_test_image(input_height, input_width, channels=1, seed=123)
    color_image_orig = create_test_image(input_height, input_width, channels=3, seed=456)

    test_cases = [
        {"name": "Grayscale Image", "image": gray_image_orig, "mode": "org"}, # Traditional 'vec' is problematic
        {"name": "Grayscale Image (Vec)", "image": gray_image_orig, "mode": "vec"},
        {"name": "Color Image", "image": color_image_orig, "mode": "org"}, # Traditional 'vec' is problematic
        {"name": "Color Image (Vec)", "image": color_image_orig, "mode": "vec"},
    ]

    results_summary = []

    for case in test_cases:
        print(f"\n--- Processing Case: {case['name']} (Mode: {case['mode']}) ---")
        img_in = case['image']
        current_mode = case['mode']

        res_traditional, res_opt_float, res_opt_fixed = None, None, None
        time_trad, time_float, time_fixed = -1, -1, -1

        # 1. Traditional Bicubic
        if imresize_traditional:
            print("Running Traditional Bicubic...")
            try:
                start_time = time.time()
                res_traditional = imresize_traditional(img_in, output_shape=target_output_shape, method='bicubic', mode=current_mode)
                time_trad = time.time() - start_time
                print(f"  Traditional done in {time_trad:.4f}s. Output shape: {res_traditional.shape}")
            except Exception as e:
                print(f"  Error in Traditional Bicubic: {e}")
        else:
            print("Traditional bicubic not available.")

        # 2. Optimized Bicubic (Float)
        if imresize_optimized_float:
            # Optimized float version should use its own 'vec' or 'org' mode handling, which is robust
            print(f"Running Optimized Bicubic (Float) with mode='{current_mode}'...")
            try:
                start_time = time.time()
                res_opt_float = imresize_optimized_float(img_in, output_shape=target_output_shape, method='bicubic_hw_friendly', mode=current_mode)
                time_float = time.time() - start_time
                print(f"  Optimized Float done in {time_float:.4f}s. Output shape: {res_opt_float.shape}")
            except Exception as e:
                print(f"  Error in Optimized Bicubic (Float): {e}")
        else:
            print("Optimized bicubic (float) not available.")

        # 3. Optimized Bicubic (Fixed-Point)
        if imresize_fixed_point:
            # Fixed-point version currently only implements 'vec' mode
            if current_mode == 'vec':
                print("Running Optimized Bicubic (Fixed-Point) with mode='vec'...")
                try:
                    start_time = time.time()
                    res_opt_fixed = imresize_fixed_point(img_in, output_shape=target_output_shape, method='bicubic_fixed', mode='vec')
                    time_fixed = time.time() - start_time
                    print(f"  Optimized Fixed-Point done in {time_fixed:.4f}s. Output shape: {res_opt_fixed.shape}")
                except Exception as e:
                    import traceback
                    print(f"  Error in Optimized Bicubic (Fixed-Point): {e}")
                    # traceback.print_exc() # Uncomment for detailed traceback
            else:
                print(f"  Optimized Fixed-Point: Skipping for mode='{current_mode}' (only 'vec' implemented).")
        else:
            print("Optimized bicubic (fixed-point) not available.")

        # PSNR Comparisons (using traditional as baseline)
        psnr_float_vs_trad = -1
        psnr_fixed_vs_trad = -1
        psnr_fixed_vs_float = -1

        if res_traditional is not None and res_opt_float is not None:
            psnr_float_vs_trad = calculate_psnr(res_traditional, res_opt_float)
            print(f"  PSNR (Optimized Float vs Traditional): {psnr_float_vs_trad:.2f} dB")

        if res_traditional is not None and res_opt_fixed is not None:
            psnr_fixed_vs_trad = calculate_psnr(res_traditional, res_opt_fixed)
            print(f"  PSNR (Optimized Fixed-Point vs Traditional): {psnr_fixed_vs_trad:.2f} dB")

        if res_opt_float is not None and res_opt_fixed is not None:
            psnr_fixed_vs_float = calculate_psnr(res_opt_float, res_opt_fixed) # Compare fixed to its float ideal
            print(f"  PSNR (Optimized Fixed-Point vs Optimized Float): {psnr_fixed_vs_float:.2f} dB")

        results_summary.append({
            "Case": case['name'],
            "Mode": current_mode,
            "Time Traditional (s)": f"{time_trad:.4f}" if time_trad != -1 else "N/A",
            "Time Opt Float (s)": f"{time_float:.4f}" if time_float != -1 else "N/A",
            "Time Opt Fixed (s)": f"{time_fixed:.4f}" if time_fixed != -1 else "N/A",
            "PSNR OptFloat_vs_Trad (dB)": f"{psnr_float_vs_trad:.2f}" if psnr_float_vs_trad != -1 else "N/A",
            "PSNR OptFixed_vs_Trad (dB)": f"{psnr_fixed_vs_trad:.2f}" if psnr_fixed_vs_trad != -1 else "N/A",
            "PSNR OptFixed_vs_OptFloat (dB)": f"{psnr_fixed_vs_float:.2f}" if psnr_fixed_vs_float != -1 else "N/A",
        })

    print("\n\n--- Overall Results Summary ---")
    # Basic print for now, can be formatted into a table later if needed
    header = list(results_summary[0].keys())
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row_data in results_summary:
        print("| " + " | ".join(str(row_data[h]) for h in header) + " |")

    print("\nComparison script finished.")
