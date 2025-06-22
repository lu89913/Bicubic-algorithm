from __future__ import print_function
import numpy as np
from math import ceil, floor

# --- Helper functions (mostly from traditional_bicubic.py) ---
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

def triangle(x): # Bilinear kernel
    x = np.array(x).astype(np.float64)
    lessthanzero = np.logical_and((x>=-1),x<0)
    greaterthanzero = np.logical_and((x<=1),x>=0)
    f = np.multiply((x+1),lessthanzero) + np.multiply((1-x),greaterthanzero)
    return f

# --- Optimized Bicubic Kernel (Hardware-Friendly Coefficients, Float Arithmetic) ---
def hardware_friendly_cubic(x_float):
    """
    Hardware-friendly bicubic kernel using float arithmetic.
    Coefficients are expressed as fractions (e.g., 1.5 as 3/2)
    to show the mathematical equivalence and path to hardware optimization.
    """
    absx = np.absolute(x_float)
    absx2 = absx * absx
    absx3 = absx2 * absx

    f = np.zeros_like(x_float)

    # Condition for |x| <= 1
    # Original: 1.5*|x|^3 - 2.5*|x|^2 + 1
    # Equivalent: (3*|x|^3 - 5*|x|^2 + 2) / 2
    cond1 = (absx <= 1)
    num1 = (3 * absx3) - (5 * absx2) + 2
    f_cond1 = num1 / 2.0

    # Condition for 1 < |x| <= 2
    # Original: -0.5*|x|^3 + 2.5*|x|^2 - 4*|x| + 2
    # Equivalent: (-|x|^3 + 5*|x|^2 - 8*|x| + 4) / 2
    cond2 = (1 < absx) & (absx <= 2)
    num2 = (-absx3) + (5 * absx2) - (8 * absx) + 4
    f_cond2 = num2 / 2.0

    f = np.where(cond1, f_cond1, f)
    f = np.where(cond2, f_cond2, f)

    return f

def contributions(in_length, out_length, scale, kernel, k_width):
    if scale < 1: # Shrink
        h = lambda x_param: scale * kernel(scale * x_param)
        kernel_width_eff = 1.0 * k_width / scale
    else: # Enlarge
        h = kernel
        kernel_width_eff = k_width

    x_coords = np.arange(1, out_length + 1).astype(np.float64)
    # Position of output pixel centers in input pixel coordinates (1-based)
    u = x_coords / scale + 0.5 * (1 - 1 / scale)

    # Index of the leftmost contributing input pixel for each output pixel (0-based)
    left = np.floor(u - kernel_width_eff / 2)

    # Number of taps (coefficients) for the kernel - reverted to original logic for consistency
    P = int(ceil(kernel_width_eff)) + 2

    # Matrix of input pixel indices (0-based) for each output pixel
    # Each row corresponds to an output pixel, columns are contributing input pixel indices
    # Reverted to original logic: ind = left_expanded + np.arange(P) - 1
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1
    indices = ind.astype(np.int32)

    # Calculate weights - reverted to original logic for consistency
    # weights = h(distance from output_pixel_center_in_input_coords to input_pixel_center)
    # The original formula was: h(np.expand_dims(u, axis=1) - indices.astype(np.float64) - 1)
    # This implies a specific definition of distance relative to the 0-indexed `indices`.
    weights = h(np.expand_dims(u, axis=1) - indices.astype(np.float64) - 1)

    sum_weights = np.sum(weights, axis=1, keepdims=True)
    weights = np.divide(weights, sum_weights, out=np.zeros_like(weights), where=sum_weights != 0)

    # Mirror padding for indices
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]

    # Filter out columns that don't contribute (all zero weights for that input pixel index column)
    # This is an optimization if P is set too large or kernel is sparse.
    # For a dense kernel like bicubic with P=4, all columns should ideally contribute.
    # However, to be safe and match original logic if needed:
    # valid_cols_mask = np.any(weights, axis=0)
    # if np.any(valid_cols_mask):
    #     weights = weights[:, valid_cols_mask]
    #     indices = indices[:, valid_cols_mask]
    # For now, assuming P is correctly set to kernel_width (e.g., 4 for bicubic),
    # so all P columns are generally used. The original P definition (ceil(kw)+2)
    # might have required this filtering more. Let's keep it simple if P=k_width.
    # If k_width=4, P=4.

    return weights, indices

# --- Resizing functions (optimized for N-D) ---
def imresizemex_optimized(inimg, weights, indices, dim):
    # This is the loop-based version, kept for compatibility/reference
    inimg_c = inimg.astype(np.float64)
    out_shape = list(inimg_c.shape)
    out_shape[dim] = weights.shape[0]
    outimg = np.zeros(out_shape, dtype=np.float64)

    other_dims_shape = list(inimg_c.shape)
    other_dims_shape.pop(dim)

    # Create an iterator for all other dimensions
    # E.g., if inimg is (H,W,C) and dim=0 (H), other_dims are (W,C)
    # if dim=1 (W), other_dims are (H,C)

    # Reverting to the exact structure of imresizemex from traditional_bicubic.py
    # to ensure maximum consistency for org mode comparison.
    # This version might only correctly handle 2D images directly if not part of a channel iteration.
    # However, imresize_optimized_float now iterates channels for 3D images, passing 2D slices here.
    if dim == 0:
        # Assuming inimg_c is a 2D slice here
        for i_img_other_dim in range(inimg_c.shape[1]): # Iterate over columns
            for i_w in range(weights.shape[0]): # Iterate over output rows
                w = weights[i_w, :]
                ind = indices[i_w, :]
                # In traditional, there was a .astype(np.float64) here, but inimg_c is already float64
                # Also, .squeeze(axis=0) was used, which implies im_slice might have been column vector.
                # For 2D slice inimg_c[ind, i_img_other_dim] is 1D array if ind is 1D array.
                im_slice = inimg_c[ind, i_img_other_dim]
                outimg[i_w, i_img_other_dim] = np.sum(im_slice * w) # Original was np.sum(np.multiply(np.squeeze(im_slice), w.T))
    elif dim == 1:
        # Assuming inimg_c is a 2D slice here
        for i_img_other_dim in range(inimg_c.shape[0]): # Iterate over rows
            for i_w in range(weights.shape[0]): # Iterate over output columns
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg_c[i_img_other_dim, ind]
                outimg[i_img_other_dim, i_w] = np.sum(im_slice * w)
    else:
        # Should not happen if called with 2D slices and dim is 0 or 1
        raise ValueError(f"imresizemex_optimized received unexpected dim: {dim} for image shape {inimg_c.shape}")


    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg

def imresizevec_optimized(inimg, weights, indices, dim):
    inimg_float = inimg.astype(np.float64)
    w_shape = weights.shape # (out_length_dim, num_coeffs)

    if dim == 0: # Interpolating along columns (dimension 0)
        gathered_pixels = inimg_float[indices]
        reshaped_weights = weights.reshape(w_shape[0], w_shape[1], *((1,)*(inimg_float.ndim - 1)))
        outimg = np.sum(gathered_pixels * reshaped_weights, axis=1)
    elif dim == 1:
        permute_order = np.roll(np.arange(inimg_float.ndim), -dim)
        img_perm = np.transpose(inimg_float, permute_order)
        gathered_pixels = img_perm[indices]
        reshaped_weights = weights.reshape(w_shape[0], w_shape[1], *((1,)*(img_perm.ndim - 1)))
        interpolated_permuted = np.sum(gathered_pixels * reshaped_weights, axis=1)
        inv_permute_order = np.argsort(permute_order)
        outimg = np.transpose(interpolated_permuted, inv_permute_order)
    else:
        raise ValueError(f"Invalid dimension '{dim}' for interpolation in imresizevec_optimized. Must be 0 or 1.")

    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg

def resizeAlongDim_optimized(A, dim, weights, indices, mode="vec"):
    if mode == "org":
        out = imresizemex_optimized(A, weights, indices, dim)
    else: # Default to 'vec'
        out = imresizevec_optimized(A, weights, indices, dim)
    return out

def imresize_optimized_float(I, scalar_scale=None, method='bicubic_hw_friendly', output_shape=None, mode="vec"):
    if method == 'bicubic_hw_friendly':
        kernel = hardware_friendly_cubic
        kernel_width = 4.0
    elif method == 'bilinear': # Kept for completeness
        kernel = triangle
        kernel_width = 2.0
    else:
        raise ValueError(f'Unidentified kernel method: {method}. Use "bicubic_hw_friendly" or "bilinear".')

    if scalar_scale is not None and output_shape is not None:
        raise ValueError('Either scalar_scale OR output_shape should be defined')
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
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
    order = np.argsort(scale_np) # Process dimension with smaller zoom factor first

    # Calculate weights and indices once, as they are independent of channel values
    weights_all_dims = []
    indices_all_dims = []
    # Calculate weights once using the shape of the first channel (if color) or the image itself (if grayscale)
    # This assumes all channels have the same spatial dimensions.
    ref_shape_for_weights = I.shape[:2] # Use H, W from original image for weight calculation

    for k_dim_idx in range(2): # Iterate over the two spatial dimensions
        dim_size_in = ref_shape_for_weights[k_dim_idx] # Use H or W from original image
        dim_size_out = output_size[k_dim_idx]
        dim_scale = scale[k_dim_idx]
        w, ind = contributions(dim_size_in, dim_size_out, dim_scale, kernel, kernel_width)
        weights_all_dims.append(w)
        indices_all_dims.append(ind)

    if I.ndim > 2:
        num_channels = I.shape[2]
        B_channels = []
        for i_chan in range(num_channels):
            I_channel = I[..., i_chan]
            B_ch = np.copy(I_channel).astype(np.float64)
            for k_pass in range(2): # Two passes for separable interpolation
                dim_to_process = order[k_pass]
                current_weights = weights_all_dims[dim_to_process]
                current_indices = indices_all_dims[dim_to_process]
                # resizeAlongDim_optimized should handle 2D B_ch correctly
                B_ch = resizeAlongDim_optimized(B_ch, dim_to_process, current_weights, current_indices, mode)
            B_channels.append(B_ch)
        B_final = np.stack(B_channels, axis=2)
    elif I.ndim == 2: # Grayscale image
        B = np.copy(I).astype(np.float64)
        # Mimic traditional_bicubic's flag2D behavior: expand, process, squeeze
        B_expanded = np.expand_dims(B, axis=2) # HxWx1

        processed_expanded = B_expanded # Initialize with correct shape for loop
        for k_pass in range(2):
            dim_to_process = order[k_pass]
            current_weights = weights_all_dims[dim_to_process]
            current_indices = indices_all_dims[dim_to_process]

            # Pass the 2D slice to resizeAlongDim_optimized, as imresizemex_optimized expects 2D
            current_slice_2D = np.squeeze(processed_expanded, axis=2)
            processed_slice_2D = resizeAlongDim_optimized(current_slice_2D, dim_to_process, current_weights, current_indices, mode)
            processed_expanded = np.expand_dims(processed_slice_2D, axis=2) # Re-expand for next pass or final squeeze

        B_final = np.squeeze(processed_expanded, axis=2)
    else:
        raise ValueError(f"Input image must be 2D or 3D, got {I.ndim}D")

    # Type conversion back to uint8 if original was uint8 is handled within resizeAlongDim_optimized's sub-functions.
    # (which in turn call imresizemex_optimized or imresizevec_optimized that handle it)
    return B_final


if __name__ == '__main__':
    print("Testing optimized_bicubic_float.py")

    img_gray = np.array([
        [10, 20, 30, 40],
        [50, 60, 70, 80],
        [90, 100, 110, 120],
        [130, 140, 150, 160]
    ], dtype=np.uint8)

    img_color_ch1 = np.array([[10,20],[50,60]], dtype=np.uint8)
    img_color_ch2 = np.array([[30,40],[70,80]], dtype=np.uint8)
    img_color_ch3 = np.array([[15,25],[55,65]], dtype=np.uint8)
    img_color = np.stack((img_color_ch1, img_color_ch2, img_color_ch3), axis=2)

    print(f"Original grayscale shape: {img_gray.shape}")
    resized_gray_vec = imresize_optimized_float(img_gray, output_shape=(6, 6), method='bicubic_hw_friendly', mode='vec')
    print(f"Resized grayscale (vec mode) shape: {resized_gray_vec.shape}")
    # print("Resized_gray_vec:\n", resized_gray_vec)

    resized_gray_org = imresize_optimized_float(img_gray, output_shape=(6, 6), method='bicubic_hw_friendly', mode='org')
    print(f"Resized grayscale (org mode) shape: {resized_gray_org.shape}")
    # print("Resized_gray_org:\n", resized_gray_org)
    # Should be identical if both vec and org are correct
    # print("Difference between vec and org for gray:", np.sum(np.abs(resized_gray_vec.astype(np.float64) - resized_gray_org.astype(np.float64))))


    print(f"\nOriginal color shape: {img_color.shape}")
    resized_color_vec = imresize_optimized_float(img_color, output_shape=(3,3), method='bicubic_hw_friendly', mode='vec')
    print(f"Resized color (vec mode) shape: {resized_color_vec.shape}")
    # print("Resized_color_vec:\n", resized_color_vec)

    resized_color_org = imresize_optimized_float(img_color, output_shape=(3,3), method='bicubic_hw_friendly', mode='org')
    print(f"Resized color (org mode) shape: {resized_color_org.shape}")
    # print("Resized_color_org:\n", resized_color_org)
    # print("Difference between vec and org for color:", np.sum(np.abs(resized_color_vec.astype(np.float64) - resized_color_org.astype(np.float64))))

    print("Basic tests in optimized_bicubic_float.py completed.")
