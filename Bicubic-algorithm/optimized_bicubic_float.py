import numpy as np
from PIL import Image
import math
from math import ceil, floor
import time # For timing

# --- Start of optimized_bicubic_float.py ---

# Global parameters for LUT
SUBPIXEL_LEVELS = 128  # Number of discrete steps for subpixel positions within a pixel
KERNEL_RADIUS = 2.0   # Bicubic kernel support is [-2, 2]
CUBIC_PARAM_A = -0.5  # Standard 'a' for bicubic, Catmull-Rom if a=-0.5

# Precomputed LUT for the cubic kernel
CUBIC_LUT = None

def _cubic_kernel_formula(x, a):
    """The actual mathematical formula for the cubic kernel."""
    x = abs(x)
    if x <= 1.0:
        return (a + 2.0) * (x**3) - (a + 3.0) * (x**2) + 1.0
    elif 1.0 < x <= 2.0:
        return a * (x**3) - 5.0 * a * (x**2) + 8.0 * a * x - 4.0 * a
    else:
        return 0.0

def precompute_bicubic_lut(subpixel_levels=SUBPIXEL_LEVELS, kernel_radius=KERNEL_RADIUS, a=CUBIC_PARAM_A):
    """
    Precomputes the LUT for the cubic kernel C(x).
    The LUT stores values for x in [0, kernel_radius].
    """
    global CUBIC_LUT
    lut_size = int(ceil(kernel_radius * subpixel_levels)) + 1
    CUBIC_LUT = np.zeros(lut_size, dtype=np.float64)
    for i in range(lut_size):
        x = i / float(subpixel_levels) # Ensure float division
        CUBIC_LUT[i] = _cubic_kernel_formula(x, a)
    # print(f"Cubic LUT created with size {lut_size} for levels={subpixel_levels}, radius={kernel_radius}")

def get_cubic_from_lut(x_abs, lut=None, subpixel_levels=SUBPIXEL_LEVELS, kernel_radius=KERNEL_RADIUS):
    """
    Gets the cubic kernel value from the precomputed LUT for a given absolute distance |x|.
    Uses nearest-neighbor lookup.
    """
    if lut is None:
        lut = CUBIC_LUT

    if x_abs > kernel_radius:
        return 0.0

    lut_index = min(int(round(x_abs * subpixel_levels)), len(lut) - 1)
    return lut[lut_index]

def contributions_lut(in_length, out_length, scale, k_width, subpixel_levels_for_lut, lut_array):
    """
    Calculate contributions using a direct vectorized LUT lookup.
    k_width: The conceptual width of the kernel (e.g., 4.0 for bicubic).
    subpixel_levels_for_lut: The number of subpixel levels used to create the LUT.
    lut_array: The precomputed LUT (e.g., CUBIC_LUT).
    """
    kernel_radius = k_width / 2.0 # e.g. 2.0 for bicubic

    if scale < 1: # Downsampling
        effective_kernel_width = k_width / scale
    else: # Upsampling
        effective_kernel_width = k_width

    x_coords = np.arange(1, out_length + 1).astype(np.float64)
    u = x_coords / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - effective_kernel_width / 2.0)

    P = int(ceil(effective_kernel_width)) + 2
    ind_1based = np.expand_dims(left, axis=1) + (np.arange(P) - 1)

    distances = np.expand_dims(u, axis=1) - ind_1based.astype(np.float64)

    # Directly vectorized LUT lookup
    abs_distances = np.abs(distances)

    # Map distances to LUT indices
    # lut_indices shape will be same as distances
    lut_indices = np.round(abs_distances * subpixel_levels_for_lut).astype(int)

    # Clip indices to be within LUT bounds
    np.clip(lut_indices, 0, len(lut_array) - 1, out=lut_indices)

    weights = lut_array[lut_indices]

    # Handle scaling for downsampling case
    if scale < 1:
        weights = weights * scale # Apply scaling factor

    # Set weights to 0 for distances outside kernel support.
    # The LUT itself should ideally have zeros for x > radius,
    # but an explicit check here is safer if distances can map unpredictably due to 'scale * x' for downsampling.
    # Effective distances for LUT lookup when downsampling are abs(scale * distances)
    effective_abs_distances_for_lut = abs_distances
    if scale < 1:
        effective_abs_distances_for_lut = np.abs(distances * scale) # these were the x in C(scale*x)

    weights[effective_abs_distances_for_lut > kernel_radius] = 0.0

    sum_weights = np.sum(weights, axis=1, keepdims=True)
    weights = np.divide(weights, sum_weights, out=np.zeros_like(weights), where=sum_weights!=0)

    indices_0based = ind_1based.astype(np.int32) - 1

    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices_0based_mirrored = aux[np.mod(indices_0based, aux.size)]

    valid_cols_mask = np.any(weights, axis=0)
    if np.any(valid_cols_mask):
        weights = weights[:, valid_cols_mask]
        indices_0based_mirrored = indices_0based_mirrored[:, valid_cols_mask]

    return weights, indices_0based_mirrored

def deriveSizeFromScale(img_shape, scale):
    output_shape = []
    for k_dim in range(2): # Corrected variable name
        output_shape.append(int(ceil(scale[k_dim] * img_shape[k_dim])))
    return output_shape

def deriveScaleFromSize(img_shape_in, img_shape_out):
    scale = []
    for k_dim in range(2): # Corrected variable name
        scale.append(1.0 * img_shape_out[k_dim] / img_shape_in[k_dim])
    return scale

def imresizevec_opt(inimg, weights, indices_0based, dim):
    inimg_float = inimg.astype(np.float64)
    if dim == 0:
        w_reshaped_dims = [weights.shape[0], weights.shape[1]] + [1] * (inimg_float.ndim - 1)
        w_reshaped = weights.reshape(w_reshaped_dims)
        slice_obj = [slice(None)] * inimg_float.ndim
        slice_obj[0] = indices_0based
        gathered_pixels = inimg_float[tuple(slice_obj)]
        outimg = np.sum(w_reshaped * gathered_pixels, axis=1)
    elif dim == 1:
        w_reshaped_dims = [1, weights.shape[0], weights.shape[1]] + [1] * (inimg_float.ndim - 2)
        w_reshaped = weights.reshape(w_reshaped_dims)
        slice_obj = [slice(None)] * inimg_float.ndim
        slice_obj[1] = indices_0based
        gathered_pixels = inimg_float[tuple(slice_obj)]
        outimg = np.sum(w_reshaped * gathered_pixels, axis=2)
    else:
        raise ValueError(f"Invalid dimension {dim} for interpolation.")

    if inimg.dtype == np.uint8: # Ensure output type matches input for uint8
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg # Return float for float inputs

def resizeAlongDim_opt(A, dim, weights, indices_0based, mode="vec"):
    if mode != "vec":
        print("Warning: Optimized version primarily supports 'vec' mode. Using 'vec'.")
    out = imresizevec_opt(A, weights, indices_0based, dim)
    return out

def imresize_optimized_float(I, scalar_scale=None, output_shape=None, method='bicubic_lut'):
    if method != 'bicubic_lut':
        raise ValueError("This function is optimized for 'bicubic_lut' method.")

    if CUBIC_LUT is None: # Ensure LUT is precomputed if not done globally
        # print("Optimized_float: LUT not found, precomputing now...")
        precompute_bicubic_lut()

    kernel_width = KERNEL_RADIUS * 2.0

    if scalar_scale is not None and output_shape is not None:
        raise ValueError('Either scalar_scale OR output_shape should be defined')
    if scalar_scale is not None:
        try: scalar_scale = float(scalar_scale)
        except ValueError: raise ValueError('scalar_scale must be a number')
        if scalar_scale <= 0: raise ValueError('scalar_scale must be positive')
        scale = [scalar_scale, scalar_scale]
        output_size = deriveSizeFromScale(I.shape[:2], scale)
    elif output_shape is not None:
        if not (isinstance(output_shape, (list, tuple)) and len(output_shape) == 2):
            raise ValueError('output_shape must be a list or tuple of 2 elements')
        if any(s <= 0 for s in output_shape): raise ValueError('output_shape dimensions must be positive')
        scale = deriveScaleFromSize(I.shape[:2], output_shape)
        output_size = list(output_shape)
    else:
        raise ValueError('Either scalar_scale OR output_shape should be defined')

    scale_np = np.array(scale)
    order = np.argsort(scale_np)

    weights_all_dims = []
    indices_all_dims = []
    for k_dim_idx in range(2):
        dim_size_in = I.shape[k_dim_idx]
        dim_size_out = output_size[k_dim_idx]
        dim_scale = scale[k_dim_idx]

        # Call contributions_lut with the CUBIC_LUT directly
        w, ind_0based = contributions_lut(dim_size_in, dim_size_out, dim_scale,
                                          kernel_width, SUBPIXEL_LEVELS, CUBIC_LUT)
        weights_all_dims.append(w)
        indices_all_dims.append(ind_0based)

    B = np.copy(I) # Work on a copy
    for k_pass in range(2):
        dim_to_process = order[k_pass]
        current_weights = weights_all_dims[dim_to_process]
        current_indices_0based = indices_all_dims[dim_to_process]
        B = resizeAlongDim_opt(B, dim_to_process, current_weights, current_indices_0based, mode="vec")

    return B

def calculate_psnr_arrays(arr1, arr2, max_pixel_value=255):
    if arr1.shape != arr2.shape:
        # print(f"Error: Array dimensions must match for PSNR. Got {arr1.shape} and {arr2.shape}")
        return None # Or raise error
    mse = np.mean((arr1.astype(np.float64) - arr2.astype(np.float64)) ** 2)
    if mse == 0: return float('inf')
    psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse))
    return psnr

if __name__ == '__main__':
    print("Running tests for optimized_bicubic_float.py...")

    precompute_bicubic_lut(subpixel_levels=SUBPIXEL_LEVELS, kernel_radius=KERNEL_RADIUS, a=CUBIC_PARAM_A)
    # print(f"Cubic LUT precomputed with {len(CUBIC_LUT)} entries for SUBPIXEL_LEVELS={SUBPIXEL_LEVELS}.")

    # Assuming the script is run from the project root (/app)
    # and images are also in the project root.
    golden_img_path = "lena_golden_512.png"
    input_img_path = "lena_downscaled_256.png"
    # Output can also be in the root, or inside Bicubic-algorithm if preferred.
    # Let's keep output in the root for now.
    output_img_path = "lena_optimized_bicubic_float_512.png"

    try:
        from PIL import Image
    except ImportError:
        print("Pillow library is not installed. Please install it: pip install Pillow")
        exit()

    try:
        golden_img_pil = Image.open(golden_img_path).convert('L')
        input_img_pil = Image.open(input_img_path).convert('L')

        golden_array = np.array(golden_img_pil)
        input_array = np.array(input_img_pil)

        print(f"Input image shape: {input_array.shape}")
        print(f"Golden image shape: {golden_array.shape}")

        target_shape = golden_array.shape

        print("\nTesting optimized float version (LUT-based)...")

        start_time = time.time()
        resized_optimized = imresize_optimized_float(input_array, output_shape=target_shape, method='bicubic_lut')
        end_time = time.time()

        print(f"Resized optimized float shape: {resized_optimized.shape}")
        print(f"Time taken for optimized float: {end_time - start_time:.4f} seconds")

        psnr_optimized = calculate_psnr_arrays(golden_array, resized_optimized)
        if psnr_optimized is not None:
            print(f"PSNR (optimized float vs golden): {psnr_optimized:.4f} dB")

        if resized_optimized.shape == target_shape:
            Image.fromarray(resized_optimized.astype(np.uint8)).save(output_img_path)
            print(f"Saved optimized float result to {output_img_path}")

        print("\nReference PSNR from traditional_bicubic.py (vec mode) was ~34.1076 dB.")
        print("Target: PSNR for optimized version should be similar or better, and runtime significantly reduced.")

    except FileNotFoundError as e:
        print(f"Error: Could not find Lena image files. Expected at {golden_img_path} and {input_img_path}.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"An error occurred during optimized float image processing tests: {e}")
        import traceback
        traceback.print_exc()

    print("\nOptimized float bicubic script tests completed.")
