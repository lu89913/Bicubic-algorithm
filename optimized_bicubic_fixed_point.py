from __future__ import print_function
import numpy as np
from math import ceil, floor

# --- Fixed-Point Parameters ---
# W: Total bits, F: Fractional bits
# For intermediate calculations of kernel coefficients and distances
FP_W_Kernel = 16
FP_F_Kernel = 8  # x, x^2, x^3, and kernel weights. Range approx -128 to +127.996
FP_MIN_Kernel = -(2**(FP_W_Kernel - 1))
FP_MAX_Kernel = (2**(FP_W_Kernel - 1)) - 1

# For pixel data and accumulation
FP_W_Pixel = 24 # Needs to handle sum of products (e.g., 4 * (255 * Weight))
FP_F_Pixel = 8  # Pixel values (0-255) will be scaled. Weights also scaled.
FP_MIN_Pixel = -(2**(FP_W_Pixel - 1))
FP_MAX_Pixel = (2**(FP_W_Pixel - 1)) - 1

# --- Fixed-Point Helper Functions ---
def saturate(value, min_val, max_val):
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    return value

def float_to_fixed(value, F, W, signed=True):
    scaled_value = value * (2**F)
    # Round to nearest integer, then convert to int.
    # np.round can result in float e.g. 2.0, ensure it's int for bitwise ops later if any.
    int_value = int(np.round(scaled_value))

    # Saturation based on W (assuming F is part of W)
    if signed:
        min_val = -(2**(W - 1))
        max_val = (2**(W - 1)) - 1
    else: # Unsigned
        min_val = 0
        max_val = (2**W) - 1
    return saturate(int_value, min_val, max_val)

def fixed_to_float(fixed_value, F):
    return float(fixed_value) / (2**F)

def fixed_add(a_fixed, b_fixed, W, F, signed=True): # F is not used here but kept for consistency
    res = a_fixed + b_fixed
    if signed:
        min_val = -(2**(W - 1))
        max_val = (2**(W - 1)) - 1
    else:
        min_val = 0
        max_val = (2**W) - 1
    return saturate(res, min_val, max_val)

def fixed_subtract(a_fixed, b_fixed, W, F, signed=True):
    res = a_fixed - b_fixed
    if signed:
        min_val = -(2**(W - 1))
        max_val = (2**(W - 1)) - 1
    else:
        min_val = 0
        max_val = (2**W) - 1
    return saturate(res, min_val, max_val)

def fixed_multiply(a_fixed, b_fixed, W_out, F_a, F_b, F_out, signed=True):
    # Product has F_a + F_b fractional bits initially
    temp_product = np.int64(a_fixed) * np.int64(b_fixed) # Use larger type for intermediate product

    # Shift to align to F_out fractional bits
    # If F_a + F_b > F_out, we need to right shift by (F_a + F_b - F_out)
    # If F_a + F_b < F_out, we need to left shift by (F_out - (F_a + F_b))
    shift_amount = (F_a + F_b) - F_out

    if shift_amount > 0:
        # Rounding before shifting (common in DSP for better accuracy)
        # Add 0.5 of the LSB that will be shifted out
        if temp_product >= 0:
            rounded_product = temp_product + (1 << (shift_amount - 1))
        else:
            # For negative numbers, rounding towards zero means adding -0.5 effectively
            # or subtracting 0.5 from absolute before shifting.
            # Simpler: let np.round handle it before full conversion, or accept truncation for now.
            # For now, direct shift (truncation for positive, floor for negative effect)
            rounded_product = temp_product
        res_fixed = int(rounded_product >> shift_amount)
    elif shift_amount < 0:
        res_fixed = int(temp_product << (-shift_amount))
    else:
        res_fixed = int(temp_product)

    if signed:
        min_val = -(2**(W_out - 1))
        max_val = (2**(W_out - 1)) - 1
    else:
        min_val = 0
        max_val = (2**W_out) - 1
    return saturate(res_fixed, min_val, max_val)

# --- Fixed-Point Bicubic Kernel ---
def hardware_friendly_cubic_fixed_point(x_float):
    """
    Fixed-point hardware-friendly bicubic kernel.
    x_float is the floating point distance |x|.
    Returns a fixed-point weight (scaled integer).
    The output scaling is F_Kernel.
    """
    # Convert float input x to fixed-point for kernel calculations
    # x is typically in [0, 2.0]. With F_Kernel=8, 2.0 -> 512. Fits in W_Kernel=16.
    x_fixed = float_to_fixed(np.absolute(x_float), FP_F_Kernel, FP_W_Kernel)

    # absx2_fixed will have 2*FP_F_Kernel fractional bits before adjustment
    absx2_fixed_temp = np.int64(x_fixed) * np.int64(x_fixed)
    # Adjust to FP_F_Kernel fractional bits for consistency in subsequent calcs
    absx2_fixed = int(absx2_fixed_temp >> FP_F_Kernel)
    absx2_fixed = saturate(absx2_fixed, FP_MIN_Kernel, FP_MAX_Kernel)

    # absx3_fixed_temp will have FP_F_Kernel (from absx2_fixed) + FP_F_Kernel (from x_fixed) fractional bits
    absx3_fixed_temp = np.int64(absx2_fixed) * np.int64(x_fixed)
    absx3_fixed = int(absx3_fixed_temp >> FP_F_Kernel)
    absx3_fixed = saturate(absx3_fixed, FP_MIN_Kernel, FP_MAX_Kernel)

    # Coefficients as fixed-point integers (scaled by 2^FP_F_Kernel, but they are integers here)
    # Or, treat them as integers and adjust scaling at the very end.
    # Let's use integer coefficients and a final shift for the /2 factor.
    # The intermediate terms like (3 * absx3_fixed) will have FP_F_Kernel fractional bits.

    _3_fixed = 3 # Integer, effectively 3 * 2^0
    _5_fixed = 5
    _8_fixed = 8
    _2_const_fixed = float_to_fixed(2.0, FP_F_Kernel, FP_W_Kernel) # 2.0 with F_Kernel fractional bits
    _4_const_fixed = float_to_fixed(4.0, FP_F_Kernel, FP_W_Kernel) # 4.0 with F_Kernel fractional bits

    f_fixed = float_to_fixed(0.0, FP_F_Kernel, FP_W_Kernel) # Initialize output

    # Condition for |x| <= 1 (use x_float for condition, x_fixed for calculation)
    if np.absolute(x_float) <= 1:
        # num1 = (3 * absx3_fixed) - (5 * absx2_fixed) + (2.0 * 2^FP_F_Kernel)
        # All terms should have FP_F_Kernel fractional bits.
        term_x3 = fixed_multiply(float_to_fixed(_3_fixed, 0, FP_W_Kernel), absx3_fixed, FP_W_Kernel, 0, FP_F_Kernel, FP_F_Kernel)
        term_x2 = fixed_multiply(float_to_fixed(_5_fixed, 0, FP_W_Kernel), absx2_fixed, FP_W_Kernel, 0, FP_F_Kernel, FP_F_Kernel)

        num1_sum1 = fixed_subtract(term_x3, term_x2, FP_W_Kernel, FP_F_Kernel)
        num1 = fixed_add(num1_sum1, _2_const_fixed, FP_W_Kernel, FP_F_Kernel)

        # Divide by 2 (right shift by 1)
        # The result should maintain FP_F_Kernel fractional bits.
        f_fixed = int(np.round(num1 / 2.0)) # Simplest way to simulate shift with rounding for positive/negative
                                          # Or num1 >> 1 if num1 is known to be appropriately scaled for bitwise shift
        f_fixed = saturate(f_fixed, FP_MIN_Kernel, FP_MAX_Kernel)

    # Condition for 1 < |x| <= 2
    elif 1 < np.absolute(x_float) <= 2:
        # num2 = (-absx3_fixed) + (5 * absx2_fixed) - (8 * x_fixed) + (4.0 * 2^FP_F_Kernel)
        term_neg_x3 = fixed_subtract(0, absx3_fixed, FP_W_Kernel, FP_F_Kernel) # 0 - absx3
        term_5x2 = fixed_multiply(float_to_fixed(_5_fixed, 0, FP_W_Kernel), absx2_fixed, FP_W_Kernel, 0, FP_F_Kernel, FP_F_Kernel)
        term_8x = fixed_multiply(float_to_fixed(_8_fixed, 0, FP_W_Kernel), x_fixed, FP_W_Kernel, 0, FP_F_Kernel, FP_F_Kernel)

        sum1 = fixed_add(term_neg_x3, term_5x2, FP_W_Kernel, FP_F_Kernel)
        sum2 = fixed_subtract(sum1, term_8x, FP_W_Kernel, FP_F_Kernel)
        num2 = fixed_add(sum2, _4_const_fixed, FP_W_Kernel, FP_F_Kernel)

        f_fixed = int(np.round(num2 / 2.0))
        f_fixed = saturate(f_fixed, FP_MIN_Kernel, FP_MAX_Kernel)

    # else f_fixed remains 0 (for |x| > 2)
    return f_fixed # This is a fixed-point value with FP_F_Kernel fractional bits

# --- Contributions function adapted for fixed-point weights ---
def contributions_fixed_point(in_length, out_length, scale, kernel_fixed_point_func, k_width):
    # Kernel function (kernel_fixed_point_func) returns fixed-point weights (scaled integers)
    # These weights then need to be normalized. Normalization is best done in float,
    # then final weights converted back to fixed-point for multiplication with pixel data.

    if scale < 1: # Shrink
        # h_float is a wrapper that will ultimately call the fixed point kernel
        # and convert its result back to float for this lambda.
        # This lambda itself computes a float value.
        h_float = lambda x_param: scale * fixed_to_float(kernel_fixed_point_func(scale * x_param), FP_F_Kernel)
        kernel_width_eff = 1.0 * k_width / scale
    else: # Enlarge
        h_float = lambda x_param: fixed_to_float(kernel_fixed_point_func(x_param), FP_F_Kernel)
        kernel_width_eff = k_width

    x_coords = np.arange(1, out_length + 1).astype(np.float64)
    u = x_coords / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width_eff / 2)
    P = int(ceil(kernel_width_eff))

    ind = np.expand_dims(left, axis=1) + np.arange(P)
    indices = ind.astype(np.int32)

    # Calculate float weights using h_float (which internally uses fixed-point kernel)
    # The kernel_fixed_point_func expects a scalar, so we must iterate.
    distances_float_array = np.expand_dims(u, axis=1) - (indices.astype(np.float64) + 1)
    float_weights_array = np.zeros_like(distances_float_array)

    for r_idx in range(distances_float_array.shape[0]):
        for c_idx in range(distances_float_array.shape[1]):
            dist_scalar = distances_float_array[r_idx, c_idx]
            # Apply the lambda h_float logic manually for each scalar distance
            if scale < 1: # Shrink
                fixed_kernel_out = kernel_fixed_point_func(scale * dist_scalar)
                float_weights_array[r_idx, c_idx] = scale * fixed_to_float(fixed_kernel_out, FP_F_Kernel)
            else: # Enlarge
                fixed_kernel_out = kernel_fixed_point_func(dist_scalar)
                float_weights_array[r_idx, c_idx] = fixed_to_float(fixed_kernel_out, FP_F_Kernel)

    sum_float_weights = np.sum(float_weights_array, axis=1, keepdims=True)
    normalized_float_weights = np.divide(float_weights_array, sum_float_weights,
                                         out=np.zeros_like(float_weights_array), where=sum_float_weights != 0)

    # Convert normalized float weights to fixed-point for use in interpolation sum
    # These weights will multiply pixel data (scaled to FP_F_Pixel).
    # So, weights should also have FP_F_Pixel fractional bits.
    # Let's assume weights are in range like [-1, 1] or [0,1] after normalization.
    # Max weight for bicubic can be 1.0.
    fixed_point_weights = np.zeros_like(normalized_float_weights, dtype=np.int32)
    for r in range(normalized_float_weights.shape[0]):
        for c in range(normalized_float_weights.shape[1]):
            fixed_point_weights[r,c] = float_to_fixed(normalized_float_weights[r,c], FP_F_Pixel, FP_W_Pixel) # W_Pixel for range

    # Mirror padding for indices
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]

    return fixed_point_weights, indices


# --- Fixed-Point Resizing Core (Vectorized) ---
def imresizevec_fixed_point(inimg_uint8, fixed_weights, indices, dim):
    # Convert input image to fixed-point
    # Pixels 0-255. If FP_F_Pixel=8, 255 becomes 255 * 256 = 65280.
    # This fits in W_Pixel=24 (max approx 8e6 for signed).
    inimg_fixed = np.zeros_like(inimg_uint8, dtype=np.int32) # Use a type that can hold W_Pixel values
    for i in np.ndindex(inimg_uint8.shape):
        inimg_fixed[i] = float_to_fixed(float(inimg_uint8[i]), FP_F_Pixel, FP_W_Pixel, signed=False)

    w_shape = fixed_weights.shape # (out_length_dim, num_coeffs)

    # Accumulator for sum of products will have FP_F_Pixel (from image) + FP_F_Pixel (from weights) fractional bits
    # before adjustment. Sum itself does not add fractional bits if operands are aligned.
    # Output of interpolation (before converting back to uint8) should be FP_F_Pixel.

    outimg_fixed_accumulator = None # This will store the sum(products) with more frac_bits

    if dim == 0:
        gathered_pixels_fixed = inimg_fixed[indices] # (L_out, N_coeffs, D1, ...)
        # Reshape weights: (L_out, N_coeffs) -> (L_out, N_coeffs, 1, ...)
        reshaped_fixed_weights = fixed_weights.reshape(w_shape[0], w_shape[1], *((1,)*(inimg_fixed.ndim - 1)))

        # Element-wise fixed_multiply:
        # Each product: in_fixed (FP_F_Pixel) * weight_fixed (FP_F_Pixel)
        # Output of fixed_multiply should be scaled to FP_F_Pixel. (F_out = FP_F_Pixel)
        # So, intermediate product has 2*FP_F_Pixel, then shifted right by FP_F_Pixel.
        products_fixed = np.zeros_like(gathered_pixels_fixed, dtype=np.int64)
        # Correct indexing for reshaped_fixed_weights
        for i_lout in range(gathered_pixels_fixed.shape[0]):      # L_out dimension
            for i_ncoeff in range(gathered_pixels_fixed.shape[1]): # N_coeffs dimension
                # Scalar weight for this (L_out, N_coeff) pair
                current_weight_scalar = reshaped_fixed_weights[i_lout, i_ncoeff, 0] # Assuming last dims are size 1
                if inimg_fixed.ndim > 2 : # e.g. color image, reshaped_weights might be (L_out, N_coeffs, 1, 1)
                     current_weight_scalar = reshaped_fixed_weights[i_lout, i_ncoeff, 0, 0]


                for other_indices in np.ndindex(gathered_pixels_fixed.shape[2:]): # Iterate over D1, D2, ...
                    full_idx_gathered = (i_lout, i_ncoeff) + other_indices
                    pixel_val_fixed = gathered_pixels_fixed[full_idx_gathered]

                    products_fixed[full_idx_gathered] = fixed_multiply(pixel_val_fixed, current_weight_scalar,
                                                                       FP_W_Pixel, FP_F_Pixel, FP_F_Pixel, FP_F_Pixel)

        outimg_fixed_sum = np.sum(products_fixed, axis=1).astype(np.int32)
        for i in np.ndindex(outimg_fixed_sum.shape):
             outimg_fixed_sum[i] = saturate(outimg_fixed_sum[i], FP_MIN_Pixel, FP_MAX_Pixel)
        outimg_fixed_final = outimg_fixed_sum

    elif dim == 1:
        permute_order = np.roll(np.arange(inimg_fixed.ndim), -dim)
        img_perm_fixed = np.transpose(inimg_fixed, permute_order)
        gathered_pixels_fixed = img_perm_fixed[indices] # Shape (L_out, N_coeffs, D_other1, ...)
        reshaped_fixed_weights = fixed_weights.reshape(w_shape[0], w_shape[1], *((1,)*(img_perm_fixed.ndim - 1)))

        products_fixed = np.zeros_like(gathered_pixels_fixed, dtype=np.int64)
        for i_lout in range(gathered_pixels_fixed.shape[0]):
            for i_ncoeff in range(gathered_pixels_fixed.shape[1]):
                current_weight_scalar = reshaped_fixed_weights[i_lout, i_ncoeff, 0]
                if img_perm_fixed.ndim > 2 :
                     current_weight_scalar = reshaped_fixed_weights[i_lout, i_ncoeff, 0, 0]

                for other_indices in np.ndindex(gathered_pixels_fixed.shape[2:]):
                    full_idx_gathered = (i_lout, i_ncoeff) + other_indices
                    pixel_val_fixed = gathered_pixels_fixed[full_idx_gathered]

                    products_fixed[full_idx_gathered] = fixed_multiply(pixel_val_fixed, current_weight_scalar,
                                                                       FP_W_Pixel, FP_F_Pixel, FP_F_Pixel, FP_F_Pixel)

        interpolated_permuted_fixed_sum = np.sum(products_fixed, axis=1).astype(np.int32)
        for i in np.ndindex(interpolated_permuted_fixed_sum.shape):
             interpolated_permuted_fixed_sum[i] = saturate(interpolated_permuted_fixed_sum[i], FP_MIN_Pixel, FP_MAX_Pixel)

        inv_permute_order = np.argsort(permute_order)
        outimg_fixed_final = np.transpose(interpolated_permuted_fixed_sum, inv_permute_order)
    else:
        raise ValueError(f"Invalid dimension '{dim}'. Must be 0 or 1.")

    # Convert final fixed-point image back to uint8
    # Ensure outimg_uint8 has the correct number of channels if input is color
    final_spatial_shape = outimg_fixed_final.shape[:2]
    if inimg_uint8.ndim == 3:
        num_channels = inimg_uint8.shape[2]
        # Check if outimg_fixed_final has channel dimension, it should from transpose
        if outimg_fixed_final.ndim != 3 or outimg_fixed_final.shape[2] != num_channels:
             # This might happen if permute logic or sum axis was not perfectly N-D aware for channels
             # For now, assume outimg_fixed_final has the correct shape (e.g. 512,512,3)
             pass # If it's already (H,W,C)
        target_uint8_shape = final_spatial_shape + (num_channels,)
    else: # Grayscale
        target_uint8_shape = final_spatial_shape

    outimg_uint8 = np.zeros(target_uint8_shape, dtype=np.uint8)

    for i in np.ndindex(outimg_fixed_final.shape): # This iterates over all elements of outimg_fixed_final
        # The index 'i' will match the structure of outimg_fixed_final.
        # We need to map 'i' to the potentially different structure of outimg_uint8 if channels were squeezed/unsqueezed.
        # However, if outimg_fixed_final already has the target shape (e.g. 512,512,3), direct indexing is fine.
        float_val = fixed_to_float(outimg_fixed_final[i], FP_F_Pixel)
        outimg_uint8[i] = np.uint8(np.clip(np.round(float_val), 0, 255))

    return outimg_uint8


# --- Main Fixed-Point Resizing Function ---
def imresize_fixed_point(I_uint8, scalar_scale=None, method='bicubic_fixed', output_shape=None, mode="vec"):
    # For fixed point, we only implement the hardware_friendly_cubic kernel
    if method != 'bicubic_fixed':
        raise ValueError("Only 'bicubic_fixed' method is supported for imresize_fixed_point.")

    kernel_fixed = hardware_friendly_cubic_fixed_point
    kernel_width = 4.0 # Bicubic kernel width

    if scalar_scale is not None and output_shape is not None:
        raise ValueError('Either scalar_scale OR output_shape should be defined')
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        if scalar_scale <= 0: raise ValueError('scalar_scale must be positive')
        scale = [scalar_scale, scalar_scale]
        # Output shape for spatial dimensions
        output_size_spatial = deriveSizeFromScale(I_uint8.shape[:2], scale)
    elif output_shape is not None:
        if not (isinstance(output_shape, (list, tuple)) and len(output_shape) == 2):
            raise ValueError('output_shape must be a list or tuple of 2 spatial elements')
        if any(s <= 0 for s in output_shape): raise ValueError('output_shape dimensions must be positive')
        scale = deriveScaleFromSize(I_uint8.shape[:2], output_shape)
        output_size_spatial = list(output_shape)
    else:
        raise ValueError('Either scalar_scale OR output_shape should be defined')

    scale_np = np.array(scale)
    order = np.argsort(scale_np)

    weights_all_dims = []
    indices_all_dims = []
    for k_dim_idx in range(2):
        dim_size_in = I_uint8.shape[k_dim_idx]
        dim_size_out = output_size_spatial[k_dim_idx]
        dim_scale = scale[k_dim_idx]
        # Use contributions_fixed_point to get fixed-point weights
        w, ind = contributions_fixed_point(dim_size_in, dim_size_out, dim_scale, kernel_fixed, kernel_width)
        weights_all_dims.append(w)
        indices_all_dims.append(ind)

    B_fixed = np.copy(I_uint8) # Start with uint8, conversion happens inside resizeAlongDim_fixed_point

    for k_pass in range(2):
        dim_to_process = order[k_pass]
        current_fixed_weights = weights_all_dims[dim_to_process]
        current_indices = indices_all_dims[dim_to_process]

        # We need a resizeAlongDim_fixed_point that calls imresizevec_fixed_point or a fixed-point mex.
        # For now, directly call imresizevec_fixed_point as mode='vec' is primary.
        if mode == "vec":
            B_fixed = imresizevec_fixed_point(B_fixed, current_fixed_weights, current_indices, dim_to_process)
        else:
            raise NotImplementedError("Fixed-point 'org' mode (mex) is not implemented yet.")
            # B_fixed = imresizemex_fixed_point(B_fixed, current_fixed_weights, current_indices, dim_to_process)
            # Placeholder if you implement imresizemex_fixed_point

    return B_fixed


# --- Helper functions (copied from optimized_bicubic_float for structure) ---
def deriveSizeFromScale(img_shape, scale):
    output_shape = []
    for k in range(2):
        output_shape.append(int(ceil(scale[k] * img_shape[k])))
    return output_shape

def deriveScaleFromSize(img_shape_in, img_shape_out):
    scale = []
    for k in range(2):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return scale


if __name__ == '__main__':
    print("Testing optimized_bicubic_fixed_point.py")

    # Test fixed point functions
    val_f = 1.5
    fp_f_kernel = 8
    fp_w_kernel = 16
    val_fix = float_to_fixed(val_f, fp_f_kernel, fp_w_kernel)
    print(f"Float {val_f} to fixed (F={fp_f_kernel}): {val_fix} (expected {int(1.5*(2**8))}) -> {fixed_to_float(val_fix, fp_f_kernel)}")

    val_f2 = -0.75
    val_fix2 = float_to_fixed(val_f2, fp_f_kernel, fp_w_kernel)
    print(f"Float {val_f2} to fixed (F={fp_f_kernel}): {val_fix2} (expected {int(-0.75*(2**8))}) -> {fixed_to_float(val_fix2, fp_f_kernel)}")

    sum_fix = fixed_add(val_fix, val_fix2, fp_w_kernel, fp_f_kernel)
    print(f"Fixed add: {val_fix} + {val_fix2} = {sum_fix} -> {fixed_to_float(sum_fix, fp_f_kernel)} (expected {1.5-0.75})")

    mult_fix = fixed_multiply(val_fix, val_fix2, fp_w_kernel, fp_f_kernel, fp_f_kernel, fp_f_kernel)
    print(f"Fixed mult: {val_fix} * {val_fix2} (F={fp_f_kernel}) = {mult_fix} -> {fixed_to_float(mult_fix, fp_f_kernel)} (expected {1.5*-0.75})")

    # Test image resizing
    img_gray = np.array([
        [10, 20, 30, 40],
        [50, 60, 70, 80],
        [90, 100, 110, 120],
        [130, 140, 150, 160]
    ], dtype=np.uint8)

    print(f"\nOriginal grayscale shape: {img_gray.shape}")

    try:
        resized_gray_fixed = imresize_fixed_point(img_gray, output_shape=(6, 6), method='bicubic_fixed', mode='vec')
        print(f"Resized grayscale fixed-point (vec mode) shape: {resized_gray_fixed.shape}")
        # print("Resized_gray_fixed:\n", resized_gray_fixed)

        # For comparison, load the float version if available (requires other file)
        # from optimized_bicubic_float import imresize_optimized_float
        # resized_gray_float = imresize_optimized_float(img_gray, output_shape=(6,6), method='bicubic_hw_friendly', mode='vec')
        # diff = np.sum(np.abs(resized_gray_float.astype(np.float32) - resized_gray_fixed.astype(np.float32)))
        # print("Difference between fixed-point and float version:", diff)

    except Exception as e:
        import traceback
        print(f"Error during fixed-point resize: {e}")
        traceback.print_exc()

    print("Basic tests in optimized_bicubic_fixed_point.py completed.")
