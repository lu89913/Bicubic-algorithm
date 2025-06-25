import numpy as np
from PIL import Image
import math
from math import ceil, floor
import time # For timing

# --- Start of optimized_bicubic_float.py ---

# Global parameters for LUT and Fixed Point
SUBPIXEL_LEVELS = 128  # Number of discrete steps for subpixel positions within a pixel
KERNEL_RADIUS = 2.0   # Bicubic kernel support is [-2, 2]
CUBIC_PARAM_A = -0.75 # Changed to -0.75 for higher PSNR variant
LUT_FRAC_BITS = 10    # Fractional bits for fixed-point weights. Max weight is 1.0, so 1*2^10 = 1024. Min is around -0.21*1024 = -215. Fits in int16.

# Precomputed FLOAT LUT for the cubic kernel (used to derive fixed-point weights)
CUBIC_LUT_FLOAT = None

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
    Precomputes the FLOAT LUT for the cubic kernel C(x).
    This float LUT is then used as a basis for deriving fixed-point weights.
    """
    global CUBIC_LUT_FLOAT
    lut_size = int(ceil(kernel_radius * subpixel_levels)) + 1
    CUBIC_LUT_FLOAT = np.zeros(lut_size, dtype=np.float64)
    for i in range(lut_size):
        x = i / float(subpixel_levels) # Ensure float division
        CUBIC_LUT_FLOAT[i] = _cubic_kernel_formula(x, a)
    # print(f"Float Cubic LUT created with size {lut_size} for levels={subpixel_levels}, radius={kernel_radius}")

# This function might not be directly used by fixed-point version's contribution calculation,
# as that will directly use the CUBIC_LUT_FLOAT
# def get_cubic_from_lut(x_abs, lut=None, subpixel_levels=SUBPIXEL_LEVELS, kernel_radius=KERNEL_RADIUS):
#     """
#     Gets the cubic kernel value from the precomputed LUT for a given absolute distance |x|.
#     Uses nearest-neighbor lookup.
#     """
#     if lut is None:
#         lut = CUBIC_LUT_FLOAT
#
#     if x_abs > kernel_radius:
#         return 0.0
#
#     lut_index = min(int(round(x_abs * subpixel_levels)), len(lut) - 1)
#     return lut[lut_index]

def contributions_fixed_point_weights(in_length, out_length, scale, k_width,
                                      subpixel_levels_for_lut, float_lut_array, fixed_point_frac_bits):
    """
    Calculates fixed-point contribution weights.
    1. Computes float weights using float_lut_array.
    2. Normalizes float weights.
    3. Converts normalized float weights to fixed-point.

    k_width: The conceptual width of the kernel (e.g., 4.0 for bicubic).
    subpixel_levels_for_lut: The number of subpixel levels used to create the float_lut_array.
    float_lut_array: The precomputed FLOAT LUT (e.g., CUBIC_LUT_FLOAT).
    fixed_point_frac_bits: Number of fractional bits for the output fixed-point weights.
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
    lut_indices = np.round(abs_distances * subpixel_levels_for_lut).astype(int)

    # Clip indices to be within float_lut_array bounds
    np.clip(lut_indices, 0, len(float_lut_array) - 1, out=lut_indices)

    # Get float kernel values from the float LUT
    float_kernel_values = float_lut_array[lut_indices]

    # Handle scaling for downsampling case (adjusts kernel values)
    if scale < 1:
        float_kernel_values = float_kernel_values * scale

    # Set kernel values to 0 for distances effectively outside kernel support
    effective_abs_distances_for_lut = abs_distances
    if scale < 1:
        effective_abs_distances_for_lut = np.abs(distances * scale)
    float_kernel_values[effective_abs_distances_for_lut > kernel_radius] = 0.0

    # Normalize float kernel values (these are now the float weights for each tap)
    sum_float_kernel_values = np.sum(float_kernel_values, axis=1, keepdims=True)
    # Avoid division by zero if all kernel values in a row are zero
    normalized_float_weights = np.divide(float_kernel_values, sum_float_kernel_values,
                                         out=np.zeros_like(float_kernel_values),
                                         where=sum_float_kernel_values!=0)

    # Convert normalized float weights to fixed-point
    fixed_point_factor = float(1 << fixed_point_frac_bits)
    fixed_point_weights = np.round(normalized_float_weights * fixed_point_factor).astype(np.int32) # Using int32 for intermediate flexibility, can be int16 if appropriate

    indices_0based = ind_1based.astype(np.int32) - 1

    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices_0based_mirrored = aux[np.mod(indices_0based, aux.size)]

    valid_cols_mask = np.any(fixed_point_weights, axis=0) # Check non-zero fixed-point weights
    if np.any(valid_cols_mask):
        fixed_point_weights = fixed_point_weights[:, valid_cols_mask]
        indices_0based_mirrored = indices_0based_mirrored[:, valid_cols_mask]

    return fixed_point_weights, indices_0based_mirrored

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

def imresizevec_fixed_point(inimg, fixed_weights, indices_0based, dim, frac_bits):
    """
    Performs interpolation along a dimension using fixed-point weights.
    inimg: Input image (NumPy array, expected to be uint8 for image data)
    fixed_weights: NumPy array of fixed-point weights (integers)
    indices_0based: NumPy array of 0-based indices for input pixels
    dim: Dimension along which to interpolate (0 for rows, 1 for columns)
    frac_bits: Number of fractional bits used in fixed_weights
    """
    if inimg.dtype != np.uint8:
        # This function is specifically designed for uint8 image data to fixed-point processing.
        # If float input is processed, it implies a different context (e.g. intermediate float stage).
        print(f"Warning: imresizevec_fixed_point received input of type {inimg.dtype}, expected uint8.")
        # For now, we'll cast to int32 for calculations, but the premise is uint8 input.
        # If this is an intermediate stage that's already float, this fixed-point function might be misapplied.
        # Assuming it's an error or a first stage processing uint8 data.
        # Cast to something that can hold pixel values for multiplication.
        current_pixels_for_calc = inimg.astype(np.int32) # Broad type for safety if not uint8
    else:
        current_pixels_for_calc = inimg.astype(np.int32)


    # Ensure fixed_weights are also int32 for consistent calculation type if they aren't already
    fixed_weights_calc = fixed_weights.astype(np.int32)

    weighted_sum = None # To store sum of (pixel * weight)

    if dim == 0: # Interpolating along columns (affecting rows)
        # fixed_weights shape: (out_H, num_coeffs)
        # Reshape weights for broadcasting: (out_H, num_coeffs, 1, ...) for remaining dims of image
        w_reshaped_dims = [fixed_weights_calc.shape[0], fixed_weights_calc.shape[1]] + [1] * (current_pixels_for_calc.ndim - 1)
        w_reshaped = fixed_weights_calc.reshape(w_reshaped_dims)

        # Gather pixels: current_pixels_for_calc[indices_0based, :, ...]
        slice_obj = [slice(None)] * current_pixels_for_calc.ndim
        slice_obj[0] = indices_0based
        gathered_pixels = current_pixels_for_calc[tuple(slice_obj)] # Shape: (out_H, num_coeffs, W_in, [C_in])

        # Element-wise multiplication and sum over num_coeffs axis (axis 1)
        # Max product of uint8 (255) and Q10 weight (e.g. 1024 for 1.0) is ~260k.
        # Summing 4-8 such products: ~2M. Fits in int32.
        weighted_sum = np.sum(w_reshaped * gathered_pixels, axis=1, dtype=np.int32)

    elif dim == 1: # Interpolating along rows (affecting columns)
        # fixed_weights shape: (out_W, num_coeffs)
        # Reshape weights: (1, out_W, num_coeffs, 1, ...)
        w_reshaped_dims = [1, fixed_weights_calc.shape[0], fixed_weights_calc.shape[1]] + [1] * (current_pixels_for_calc.ndim - 2)
        w_reshaped = fixed_weights_calc.reshape(w_reshaped_dims)

        # Gather pixels: current_pixels_for_calc[:, indices_0based, ...]
        slice_obj = [slice(None)] * current_pixels_for_calc.ndim
        slice_obj[1] = indices_0based
        gathered_pixels = current_pixels_for_calc[tuple(slice_obj)] # Shape: (H_in, out_W, num_coeffs, [C_in])

        # Sum over num_coeffs axis (axis 2)
        weighted_sum = np.sum(w_reshaped * gathered_pixels, axis=2, dtype=np.int32)
    else:
        raise ValueError(f"Invalid dimension {dim} for interpolation.")

    # Scale back and round (division by 2**frac_bits)
    # (val + round_offset) >> frac_bits is equivalent to round(val / 2**frac_bits) for positive val
    # For potentially negative intermediate sums (if weights can be negative, which they are for bicubic),
    # simple right shift truncates towards negative infinity.
    # A common way for signed rounding: floor( (val / (2**frac_bits)) + 0.5 )
    # Or for positive/negative: (val + ( (1 << (frac_bits-1)) * np.sign(val) if val !=0 else 0) ) >> frac_bits
    # Simpler and often used: add half LSB before shift if result is positive.
    # (val + (1 << (frac_bits -1))) >> frac_bits is good for positive results.
    # Let's use np.round for clarity, then cast.
    # scaled_result_float = weighted_sum / float(1 << frac_bits)
    # scaled_rounded_result = np.round(scaled_result_float)

    # More direct fixed-point style rounding for positive numbers:
    # Add half of the divisor before integer division (right shift)
    rounding_offset = (1 << (frac_bits - 1))
    # Apply offset carefully for negative numbers if strict symmetric rounding is needed.
    # For pixel values which are positive after scaling, simpler rounding is fine.
    # If weighted_sum can be negative (due to negative weights), then:
    # scaled_result = np.where(weighted_sum >= 0,
    #                         (weighted_sum + rounding_offset) >> frac_bits,
    #                         (weighted_sum - rounding_offset + (1<<frac_bits)-1 ) >> frac_bits ) # more complex for negatives
    # A simpler way that works for positive and negative, rounding towards zero for .5 cases:
    # scaled_result = (weighted_sum + np.sign(weighted_sum) * rounding_offset if frac_bits >0 else weighted_sum) // (1 << frac_bits)
    # Or simply:
    scaled_result = np.floor((weighted_sum.astype(np.float64) / (1 << frac_bits)) + 0.5).astype(np.int32)


    # Clip to uint8 range [0, 255] as these are pixel values
    outimg_clipped = np.clip(scaled_result, 0, 255)

    # The output of this stage should be uint8 if it's the final output for that dimension,
    # or if it's an intermediate result that needs to be uint8 for the next stage.
    # If the overall function `imresize_optimized_fixed_point` processes uint8 to uint8, then this is correct.
    return outimg_clipped.astype(np.uint8)

# calculate_psnr_arrays (copied from float version, can be shared utility)
def calculate_psnr_arrays(arr1, arr2, max_pixel_value=255):
    if arr1.shape != arr2.shape:
        return None
    mse = np.mean((arr1.astype(np.float64) - arr2.astype(np.float64)) ** 2)
    if mse == 0: return float('inf')
    psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse))
    return psnr

def resizeAlongDim_fixed_point(A, dim, fixed_weights, indices_0based, frac_bits, mode="vec"):
    if mode != "vec":
        print("Warning: Fixed-point version currently supports 'vec' mode. Using 'vec'.")
    out = imresizevec_fixed_point(A, fixed_weights, indices_0based, dim, frac_bits)
    return out

def imresize_optimized_fixed_point(I, scalar_scale=None, output_shape=None, method='bicubic_fixed'):
    if method != 'bicubic_fixed':
        raise ValueError("This function is optimized for 'bicubic_fixed' method.")

    if I.dtype != np.uint8:
        raise ValueError("Input image I must be of dtype uint8 for fixed-point processing.")

    if CUBIC_LUT_FLOAT is None:
        precompute_bicubic_lut()

    kernel_width = KERNEL_RADIUS * 2.0

    if scalar_scale is not None and output_shape is not None:
        raise ValueError('Either scalar_scale OR output_shape should be defined')
    if scalar_scale is not None:
        try: scalar_scale = float(scalar_scale)
        except ValueError: raise ValueError('scalar_scale must be a number')
        if scalar_scale <= 0: raise ValueError('scalar_scale must be positive')
        scale_tuple = [scalar_scale, scalar_scale]
        output_size = deriveSizeFromScale(I.shape[:2], scale_tuple)
    elif output_shape is not None:
        if not (isinstance(output_shape, (list, tuple)) and len(output_shape) == 2):
            raise ValueError('output_shape must be a list or tuple of 2 elements')
        if any(s <= 0 for s in output_shape): raise ValueError('output_shape dimensions must be positive')
        scale_tuple = deriveScaleFromSize(I.shape[:2], output_shape)
        output_size = list(output_shape)
    else:
        raise ValueError('Either scalar_scale OR output_shape should be defined')

    scale_np_arr = np.array(scale_tuple) # Renamed to avoid conflict with 'scale' module
    order = np.argsort(scale_np_arr)

    weights_all_dims_fixed = []
    indices_all_dims = []
    for k_dim_idx in range(2):
        dim_size_in = I.shape[k_dim_idx]
        dim_size_out = output_size[k_dim_idx]
        dim_scale_val = scale_tuple[k_dim_idx]

        w_fixed, ind_0based = contributions_fixed_point_weights(
            dim_size_in, dim_size_out, dim_scale_val,
            kernel_width, SUBPIXEL_LEVELS, CUBIC_LUT_FLOAT, LUT_FRAC_BITS
        )
        weights_all_dims_fixed.append(w_fixed)
        indices_all_dims.append(ind_0based)

    B = np.copy(I)
    for k_pass in range(2):
        dim_to_process = order[k_pass]
        current_fixed_weights = weights_all_dims_fixed[dim_to_process]
        current_indices_0based = indices_all_dims[dim_to_process]
        B = resizeAlongDim_fixed_point(B, dim_to_process, current_fixed_weights, current_indices_0based, LUT_FRAC_BITS, mode="vec")

    return B

if __name__ == '__main__':
    print("Running tests for optimized_bicubic_fixed_point.py...")

    precompute_bicubic_lut()

    golden_img_path = "lena_golden_512.png"
    input_img_path = "lena_downscaled_256.png"
    output_img_path = "lena_optimized_bicubic_fixed_512.png"

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

        print(f"Input image shape: {input_array.shape}, dtype: {input_array.dtype}")
        print(f"Golden image shape: {golden_array.shape}, dtype: {golden_array.dtype}")

        target_shape = golden_array.shape

        print(f"\nTesting optimized fixed-point version (LUT_FRAC_BITS={LUT_FRAC_BITS})...")

        start_time = time.time()
        resized_fixed_point = imresize_optimized_fixed_point(input_array, output_shape=target_shape, method='bicubic_fixed')
        end_time = time.time()

        print(f"Resized fixed-point shape: {resized_fixed_point.shape}, dtype: {resized_fixed_point.dtype}")
        print(f"Time taken for optimized fixed-point: {end_time - start_time:.4f} seconds")

        psnr_fixed_point = calculate_psnr_arrays(golden_array, resized_fixed_point)
        if psnr_fixed_point is not None:
            print(f"PSNR (optimized fixed-point vs golden): {psnr_fixed_point:.4f} dB")

        if resized_fixed_point.shape == target_shape:
            Image.fromarray(resized_fixed_point.astype(np.uint8)).save(output_img_path)
            print(f"Saved optimized fixed-point result to {output_img_path}")

        print("\nReference PSNR from float version was ~34.1076 dB.")
        print("Target: PSNR for fixed-point should be close to float version.")

    except FileNotFoundError as e:
        print(f"Error: Could not find Lena image files. Expected at ./{golden_img_path} and ./{input_img_path}.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"An error occurred during optimized fixed-point image processing tests: {e}")
        import traceback
        traceback.print_exc()

    print("\nOptimized fixed-point bicubic script tests completed.")
