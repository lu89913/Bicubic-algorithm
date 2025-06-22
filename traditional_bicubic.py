from __future__ import print_function
import numpy as np
from math import ceil, floor

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

def triangle(x):
    x = np.array(x).astype(np.float64)
    lessthanzero = np.logical_and((x>=-1),x<0)
    greaterthanzero = np.logical_and((x<=1),x>=0)
    f = np.multiply((x+1),lessthanzero) + np.multiply((1-x),greaterthanzero)
    return f

def cubic(x):
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5*absx3 - 2.5*absx2 + 1, absx <= 1) + np.multiply(-0.5*absx3 + 2.5*absx2 - 4*absx + 2, (1 < absx) & (absx <= 2))
    return f

def contributions(in_length, out_length, scale, kernel, k_width):
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x_coords = np.arange(1, out_length+1).astype(np.float64)
    u = x_coords / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(ceil(kernel_width)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1 # -1 because indexing from 0
    indices = ind.astype(np.int32)

    # Calculate weights
    # The '-1' adjusts for 0-based indexing vs 1-based conceptual pixel centers
    # if u and indices are both 0-indexed coordinates.
    # Given u = x_coord/scale ..., where x_coord is 1-based, u is 1-based in input space.
    # indices are 0-based indices.
    # The original MATLAB code from which this is often derived uses 1-based indexing.
    # A common translation to Python: kernel( (output_coord_in_input_space) - (input_coord_center) )
    # output_coord_in_input_space = u
    # input_coord_center = indices + 0.5 (if indices are 0-based left edges of pixels)
    # or input_coord_center = indices + 1 (if indices are 0-based and we want to match 1-based centers)
    # The original `weights = h(np.expand_dims(u, axis=1) - indices - 1)` implies specific indexing assumptions.
    # Let's keep it as is from the user's provided code.
    weights = h(np.expand_dims(u, axis=1) - indices.astype(np.float64) - 1)

    # Normalize weights
    sum_weights = np.sum(weights, axis=1, keepdims=True)
    # Avoid division by zero, set weights to 0 if sum is 0 (though unlikely for bicubic)
    weights = np.divide(weights, sum_weights, out=np.zeros_like(weights), where=sum_weights!=0)

    # Mirror padding for indices
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]

    # Store only columns in weights and indices that have at least one non-zero weight
    # This was:
    # ind2store = np.nonzero(np.any(weights, axis=0))
    # weights = weights[:, ind2store] # This is problematic if ind2store is tuple
    # indices = indices[:, ind2store]
    # A more robust way:
    valid_cols_mask = np.any(weights, axis=0)
    if np.any(valid_cols_mask): # Check if there's at least one column with non-zero contribution
        weights = weights[:, valid_cols_mask]
        indices = indices[:, valid_cols_mask]
    # If all columns are zero (e.g. out_length is 0, or some other edge case),
    # weights/indices could become empty with shape (out_length, 0). This is acceptable.

    return weights, indices

def imresizemex(inimg, weights, indices, dim):
    # Type casting and array prep
    inimg_c = inimg.astype(np.float64)
    out_shape = list(inimg_c.shape)
    out_shape[dim] = weights.shape[0]
    outimg = np.zeros(out_shape, dtype=np.float64)

    # Interpolation
    if dim == 0: # Interpolate along columns (dim 0)
        for i_col in range(inimg_c.shape[1]): # For each column in the input image
            for i_out_row in range(weights.shape[0]): # For each output row
                w_row = weights[i_out_row, :]
                ind_row = indices[i_out_row, :]
                pixel_slice = inimg_c[ind_row, i_col]
                outimg[i_out_row, i_col] = np.sum(pixel_slice * w_row)
    elif dim == 1: # Interpolate along rows (dim 1)
        for i_row in range(inimg_c.shape[0]): # For each row in the input image
            for i_out_col in range(weights.shape[0]): # For each output column
                w_row = weights[i_out_col, :] # Note: weights.shape[0] is out_length for current dim
                ind_row = indices[i_out_col, :]
                pixel_slice = inimg_c[i_row, ind_row]
                outimg[i_row, i_out_col] = np.sum(pixel_slice * w_row)

    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg

def imresizevec(inimg, weights, indices, dim):
    # Note: This is the user's original imresizevec.
    # It has known issues with >2D images and its reshape logic.
    # It's preserved here for fidelity to the "original" version.
    # For robust N-D, a more careful transposition/reshape is needed.
    inimg_float = inimg.astype(np.float64)
    wshape = weights.shape # (out_length_dim, num_coeffs)

    if dim == 0:
        # Original: weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        # This assumes wshape has 3 dims (wshape[2]), which is not true: weights is 2D.
        # This line will cause an error if not modified.
        # For a 2D image (H,W), inimg[indices] is (H_out, N_coeffs, W)
        # weights is (H_out, N_coeffs)
        # To multiply, weights needs to be (H_out, N_coeffs, 1)
        # Then sum over axis 1 (N_coeffs)
        # If inimg is (H,W,C), inimg[indices] is (H_out, N_coeffs, W, C)
        # weights needs to be (H_out, N_coeffs, 1, 1)

        # Minimal correction to make it runnable for 2D and 3D, but still based on original structure:
        if inimg_float.ndim == 2:
             w_reshaped = weights.reshape(wshape[0], wshape[1], 1)
        elif inimg_float.ndim == 3:
             w_reshaped = weights.reshape(wshape[0], wshape[1], 1, 1)
        else:
            raise ValueError(f"imresizevec (original) unsupported input ndim: {inimg_float.ndim}")

        # Original: gathered_pixels = inimg[indices].squeeze(axis=1)
        # .squeeze(axis=1) is problematic. Advanced indexing doesn't add a singleton dim there.
        gathered_pixels = inimg_float[indices] # (H_out, N_coeffs, W) or (H_out, N_coeffs, W, C)
        outimg = np.sum(w_reshaped * gathered_pixels, axis=1)

    elif dim == 1:
        # Original: weights = weights.reshape((1, wshape[0], wshape[2], 1))
        # Similar issue with wshape[2]
        # If inimg is (H,W), inimg[:,indices] is (H, W_out, N_coeffs). Sum over axis 2.
        # weights is (W_out, N_coeffs). Needs to be (1, W_out, N_coeffs)
        # If inimg is (H,W,C), inimg[:,indices] is (H, W_out, N_coeffs, C). Sum over axis 2.
        # weights needs to be (1, W_out, N_coeffs, 1)

        if inimg_float.ndim == 2:
            w_reshaped = weights.reshape(1, wshape[0], wshape[1])
        elif inimg_float.ndim == 3:
            # This will be tricky with current indexing inimg[:, indices]
            # inimg[:, indices] for (H,W,C) -> (H, W_out, N_coeffs, C)
            # We need weights as (1, W_out, N_coeffs, 1)
            w_reshaped = weights.reshape(1, wshape[0], wshape[1], 1)
        else:
            raise ValueError(f"imresizevec (original) unsupported input ndim: {inimg_float.ndim}")

        # Original: gathered_pixels = inimg[:, indices].squeeze(axis=2)
        # Squeeze is problematic.
        gathered_pixels = inimg_float[:, indices] # (H, W_out, N_coeffs) or (H, W_out, N_coeffs, C)
        outimg = np.sum(w_reshaped * gathered_pixels, axis=2) # Sum over N_coeffs axis

    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg

def resizeAlongDim(A, dim, weights, indices, mode="vec"):
    if mode == "org":
        out = imresizemex(A, weights, indices, dim)
    else: # Default to 'vec'
        out = imresizevec(A, weights, indices, dim)
    return out

def imresize(I, scalar_scale=None, method='bicubic', output_shape=None, mode="vec"):
    if method == 'bicubic':
        kernel = cubic
        kernel_width = 4.0
    elif method == 'bilinear':
        kernel = triangle
        kernel_width = 2.0 # Bilinear kernel width is 2.0
    else:
        raise ValueError(f'Unidentified kernel method supplied: {method}')

    # Fill scale and output_size
    if scalar_scale is not None and output_shape is not None:
        raise ValueError('Either scalar_scale OR output_shape should be defined')
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        if scalar_scale <= 0:
            raise ValueError('scalar_scale must be positive')
        scale = [scalar_scale, scalar_scale]
        output_size = deriveSizeFromScale(I.shape[:2], scale) # Use first 2 dims for shape
    elif output_shape is not None:
        if not (isinstance(output_shape, (list, tuple)) and len(output_shape) == 2):
            raise ValueError('output_shape must be a list or tuple of 2 elements')
        if any(s <= 0 for s in output_shape):
            raise ValueError('output_shape dimensions must be positive')
        scale = deriveScaleFromSize(I.shape[:2], output_shape)
        output_size = list(output_shape)
    else:
        raise ValueError('Either scalar_scale OR output_shape should be defined')

    # Determine order of operations (which dimension to process first)
    scale_np = np.array(scale)
    order = np.argsort(scale_np)

    # Handle multi-channel images by processing each channel separately
    # if the core interpolation functions (imresizemex, imresizevec) are only 2D.
    # The provided imresizemex iterates over the "other" dimension, suggesting it can handle 2D slices.
    # The original imresizevec had issues with N-D. The minimally corrected one might too.

    # Let's assume the core functions will be applied to 2D slices if ndim > 2.
    if I.ndim > 2:
        num_channels = I.shape[2]
        B_channels = []
        for i_chan in range(num_channels):
            # Process each channel as a 2D image
            I_channel = I[..., i_chan]

            # Calculate weights and indices (these are independent of channel values)
            # But they depend on kernel_width which is fixed per method.
            # And on scale/output_size which are also fixed.
            # So, calculate weights once.
            if i_chan == 0: # Calculate weights only for the first channel
                weights_all_dims = []
                indices_all_dims = []
                for k_dim_idx in range(2): # Iterate over the two spatial dimensions
                    dim_size_in = I_channel.shape[k_dim_idx]
                    dim_size_out = output_size[k_dim_idx]
                    dim_scale = scale[k_dim_idx]

                    w, ind = contributions(dim_size_in, dim_size_out, dim_scale, kernel, kernel_width)
                    weights_all_dims.append(w)
                    indices_all_dims.append(ind)

            B_ch = np.copy(I_channel)
            for k_pass in range(2): # Two passes for separable interpolation
                dim_to_process = order[k_pass]
                current_weights = weights_all_dims[dim_to_process]
                current_indices = indices_all_dims[dim_to_process]
                B_ch = resizeAlongDim(B_ch, dim_to_process, current_weights, current_indices, mode)
            B_channels.append(B_ch)

        # Stack channels back
        B = np.stack(B_channels, axis=2)

    else: # Grayscale image (2D)
        weights_all_dims = []
        indices_all_dims = []
        for k_dim_idx in range(2): # Iterate over the two spatial dimensions
            dim_size_in = I.shape[k_dim_idx]
            dim_size_out = output_size[k_dim_idx]
            dim_scale = scale[k_dim_idx]

            w, ind = contributions(dim_size_in, dim_size_out, dim_scale, kernel, kernel_width)
            weights_all_dims.append(w)
            indices_all_dims.append(ind)

        B = np.copy(I)
        # The original code had a flag2D and expand_dims/squeeze for 2D inputs.
        # This was to make it pseudo-3D for resizeAlongDim if it expected 3D.
        # My imresizemex and corrected imresizevec should handle 2D inputs directly.
        # So, removing the expand/squeeze for now to simplify, assuming resizeAlongDim works for 2D.
        # If resizeAlongDim (specifically original imresizevec) fails for 2D, this might need revisit.
        # The original logic was:
        # flag2D = False
        # if B.ndim == 2:
        #    B = np.expand_dims(B, axis=2)
        #    flag2D = True
        #
        # (after loop)
        # if flag2D:
        #    B = np.squeeze(B, axis=2)
        #
        # For now, let's assume B (2D) is passed as is.
        for k_pass in range(2): # Two passes for separable interpolation
            dim_to_process = order[k_pass]
            current_weights = weights_all_dims[dim_to_process]
            current_indices = indices_all_dims[dim_to_process]
            B = resizeAlongDim(B, dim_to_process, current_weights, current_indices, mode)

    # Final type conversion (if original was uint8)
    # This is handled inside imresizemex/imresizevec, so B should already be correct type.
    # However, if B was float initially, it should remain float.
    # The functions return uint8 if input was uint8.
    return B


def convertDouble2Byte(I):
    # This function assumes I is a float array in range [0.0, 1.0]
    B = np.clip(I, 0.0, 1.0)
    B = 255*B
    return np.around(B).astype(np.uint8)

if __name__ == '__main__':
    # A simple test case
    print("Testing traditional_bicubic.py")

    # Create a dummy 2D image (grayscale)
    img_gray = np.array([
        [10, 20, 30, 40],
        [50, 60, 70, 80],
        [90, 100, 110, 120],
        [130, 140, 150, 160]
    ], dtype=np.uint8)

    # Create a dummy 3D image (color)
    img_color_ch1 = np.array([
        [10, 20], [50, 60]
    ], dtype=np.uint8)
    img_color_ch2 = np.array([
        [30, 40], [70, 80]
    ], dtype=np.uint8)
    img_color_ch3 = np.array([
        [15, 25], [55, 65]
    ], dtype=np.uint8)
    img_color = np.stack((img_color_ch1, img_color_ch2, img_color_ch3), axis=2)


    print(f"Original grayscale shape: {img_gray.shape}")
    resized_gray_org = imresize(img_gray, output_shape=(6, 6), method='bicubic', mode='org')
    print(f"Resized grayscale (org mode) shape: {resized_gray_org.shape}")
    # print("Resized_gray_org:\n", resized_gray_org)

    # The original imresizevec has issues, especially with its wshape[2] access.
    # I've put minimal fixes, let's see if it runs for vec mode.
    try:
        resized_gray_vec = imresize(img_gray, output_shape=(6, 6), method='bicubic', mode='vec')
        print(f"Resized grayscale (vec mode) shape: {resized_gray_vec.shape}")
        # print("Resized_gray_vec:\n", resized_gray_vec)
    except Exception as e:
        print(f"Error running vec mode for grayscale: {e}")

    print(f"\nOriginal color shape: {img_color.shape}")
    resized_color_org = imresize(img_color, output_shape=(3,3), method='bicubic', mode='org')
    print(f"Resized color (org mode) shape: {resized_color_org.shape}")
    # print("Resized_color_org:\n", resized_color_org)

    try:
        resized_color_vec = imresize(img_color, output_shape=(3,3), method='bicubic', mode='vec')
        print(f"Resized color (vec mode) shape: {resized_color_vec.shape}")
        # print("Resized_color_vec:\n", resized_color_vec)
    except Exception as e:
        print(f"Error running vec mode for color: {e}")

    # Test bilinear
    resized_gray_bilinear = imresize(img_gray, output_shape=(6,6), method='bilinear', mode='org')
    print(f"\nResized grayscale (bilinear, org mode) shape: {resized_gray_bilinear.shape}")

    print("\nTest with scalar_scale:")
    resized_gray_scalar = imresize(img_gray, scalar_scale=1.5, method='bicubic', mode='org')
    print(f"Resized grayscale (scalar_scale=1.5, org mode) shape: {resized_gray_scalar.shape} (Expected (6,6))")
    assert resized_gray_scalar.shape == (6,6)

    print("Basic tests in traditional_bicubic.py completed.")
