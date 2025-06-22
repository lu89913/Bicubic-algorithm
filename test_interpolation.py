import numpy as np
import time
from image_interpolation import imresize # Import from our local file
from skimage.metrics import peak_signal_noise_ratio

def create_test_image(height, width, channels=1):
    """Generates a simple test image with a gradient pattern."""
    if channels == 1:
        row_ramp = np.linspace(0, 255, width, dtype=np.uint8).reshape(1, width)
        col_ramp = np.linspace(0, 255, height, dtype=np.uint8).reshape(height, 1)
        img = np.sqrt(row_ramp**2 + col_ramp**2).astype(np.float32) # Create a radial-like gradient
        img = (img / np.max(img)) * 255 # Normalize to 0-255
        return img.astype(np.uint8)
    elif channels == 3:
        img = np.zeros((height, width, channels), dtype=np.uint8)
        # Create different patterns for each channel for variety
        for i in range(channels):
            # Simple linear gradients for each channel, offset
            base_val = (i * 60) % 256
            row_ramp = np.linspace(base_val, (base_val + 195) % 256, width, dtype=np.uint8).reshape(1, width)
            col_ramp = np.linspace(base_val, (base_val + 195) % 256, height, dtype=np.uint8).reshape(height, 1)
            channel_img = ((row_ramp + col_ramp)/2).astype(np.uint8)
            img[:,:,i] = channel_img
        return img
    else:
        raise ValueError(f"Unsupported number of channels: {channels}")


def calculate_psnr(img_true, img_test, data_range=255):
    """Calculates PSNR between two images."""
    # skimage's psnr expects inputs to be ndarray.
    # It calculates PSNR for float images in range [0,1] or [0,255] etc.
    # by default, data_range is max(img_true) - min(img_true)
    # For uint8 images, data_range=255 is standard.
    if img_true.dtype == np.uint8 and img_test.dtype == np.uint8:
        return peak_signal_noise_ratio(img_true, img_test, data_range=data_range)
    else: # Assuming float images, potentially in [0,1] or other ranges
        # If they are float, ensure data_range is appropriate.
        # If they were converted from uint8 for processing, 255 is still the ref.
        return peak_signal_noise_ratio(img_true.astype(np.float64), img_test.astype(np.float64), data_range=data_range)


if __name__ == "__main__":
    input_height, input_width = 256, 256
    output_height, output_width = 512, 512
    output_shape = (output_height, output_width)

    print(f"Generating a {input_height}x{input_width} grayscale test image...")
    gray_image = create_test_image(input_height, input_width, channels=1)

    print(f"Generating a {input_height}x{input_width}x3 color test image...")
    color_image = create_test_image(input_height, input_width, channels=3)

    images_to_test = {
        "grayscale": gray_image,
        "color": color_image
    }

    interpolation_methods = {
        "bicubic_traditional": "bicubic",
        "bicubic_hw_friendly": "bicubic_hw_friendly"
    }

    results = {}

    for img_type, test_image in images_to_test.items():
        print(f"\n--- Testing {img_type} image ({test_image.shape}) ---")

        # Store results for comparison
        method_outputs = {}

        for method_name, method_code in interpolation_methods.items():
            print(f"Performing {method_name} interpolation to {output_shape}...")
            start_time = time.time()

            # Run interpolation using 'vec' mode as it's generally faster
            # and was the focus of improvements in image_interpolation.py
            resized_image = imresize(test_image, output_shape=output_shape, method=method_code, mode="vec")

            end_time = time.time()
            duration = end_time - start_time

            method_outputs[method_name] = resized_image
            results[(img_type, method_name)] = {
                "time": duration,
                "output_shape": resized_image.shape
            }
            print(f"{method_name} completed in {duration:.4f} seconds. Output shape: {resized_image.shape}")

        # Compare HW-friendly to traditional bicubic
        if "bicubic_traditional" in method_outputs and "bicubic_hw_friendly" in method_outputs:
            img_trad = method_outputs["bicubic_traditional"]
            img_hw = method_outputs["bicubic_hw_friendly"]

            if img_trad.shape != img_hw.shape:
                print("Error: Output shapes do not match!")
                print(f"  Traditional: {img_trad.shape}, HW-Friendly: {img_hw.shape}")
            else:
                # Calculate PSNR of HW-friendly vs Traditional
                # High PSNR indicates they are very similar.
                # For uint8 images, direct comparison is fine.
                # For float images that were uint8, they should also be very close.
                # The data_range should be 255 as original data was uint8.
                psnr_val = calculate_psnr(img_trad, img_hw, data_range=255)
                results[(img_type, "comparison")] = {"psnr_hw_vs_trad": psnr_val}
                print(f"PSNR between HW-Friendly and Traditional Bicubic: {psnr_val:.2f} dB")

                # Also calculate sum of absolute differences
                abs_diff = np.sum(np.abs(img_trad.astype(np.float64) - img_hw.astype(np.float64)))
                print(f"Sum of absolute differences: {abs_diff}")


    print("\n--- Summary of Results ---")
    for key, value in results.items():
        img_type, method_info = key
        if method_info == "comparison":
            print(f"Image Type: {img_type}, Metric: PSNR (HW vs Trad), Value: {value['psnr_hw_vs_trad']:.2f} dB")
        else:
            print(f"Image Type: {img_type}, Method: {method_info}, Time: {value['time']:.4f}s, Output Shape: {value['output_shape']}")

    # Example of how to test with scalar_scale if needed
    print("\n--- Scalar Scale Test (Grayscale) ---")
    scale_factor = 2.0
    print(f"Scaling grayscale by {scale_factor}x using bicubic_hw_friendly...")
    resized_scalar = imresize(gray_image, scalar_scale=scale_factor, method='bicubic_hw_friendly', mode="vec")
    print(f"Output shape: {resized_scalar.shape}")
    expected_shape = (int(input_height * scale_factor), int(input_width * scale_factor))
    assert resized_scalar.shape == expected_shape, \
        f"Shape mismatch! Got {resized_scalar.shape}, expected {expected_shape}"

    print("\nTest script finished.")
