from __future__ import print_function
import numpy as np
from math import ceil, floor

# Original functions from the user
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
    f = np.multiply(1.5*absx3 - 2.5*absx2 + 1, absx <= 1) + \
        np.multiply(-0.5*absx3 + 2.5*absx2 - 4*absx + 2, (1 < absx) & (absx <= 2))
    return f

# New hardware-friendly cubic kernel
def hardware_friendly_cubic(x_float):
    """
    Hardware-friendly bicubic kernel.
    The coefficients are expressed as fractions to facilitate hardware
    implementation using shifts and additions, rather than general multipliers.
    Mathematically, this kernel is equivalent to the standard cubic kernel
    if floating-point precision is maintained.
    """
    absx = np.absolute(x_float)
    absx2 = absx * absx
    absx3 = absx2 * absx

    f = np.zeros_like(x_float)

    # Condition for |x| <= 1
    # Original: 1.5*|x|^3 - 2.5*|x|^2 + 1
    # Equivalent: (3*|x|^3 - 5*|x|^2 + 2) / 2
    cond1 = (absx <= 1)
    # Numerator for cond1, computed using element-wise operations if absx is an array
    num1 = (3 * absx3) - (5 * absx2) + 2
    f_cond1 = num1 / 2.0

    # Condition for 1 < |x| <= 2
    # Original: -0.5*|x|^3 + 2.5*|x|^2 - 4*|x| + 2
    # Equivalent: (-|x|^3 + 5*|x|^2 - 8*|x| + 4) / 2
    cond2 = (1 < absx) & (absx <= 2)
    # Numerator for cond2
    num2 = (-absx3) + (5 * absx2) - (8 * absx) + 4
    f_cond2 = num2 / 2.0

    # Apply conditions using np.where for clarity and correctness with arrays
    f = np.where(cond1, f_cond1, f)
    f = np.where(cond2, f_cond2, f)

    return f

def contributions(in_length, out_length, scale, kernel, k_width):
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length+1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(ceil(kernel_width)) + 2 # As in original code
    #P = int(np.ceil(kernel_width)) + 2 # Make sure ceil is from numpy if it operates on arrays. Here kernel_width is scalar.

    # Corrected P to be consistent with typical 4-point kernel for bicubic (width 4)
    # For a kernel width of 4, we need 4 contributing input pixels.
    # The number of points P should be related to kernel_width.
    # If kernel_width = 4, P is typically 4.
    # The original code uses ceil(kernel_width) + 2, which for k_width=4 gives 6.
    # This might be to handle edge cases or a specific interpretation.
    # Let's stick to the original P for now to maintain consistency with user's code.

    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1 # -1 because indexing from 0
    indices = ind.astype(np.int32)

    # Weights calculation
    # u_expanded = np.expand_dims(u, axis=1)
    # dists = u_expanded - indices -1 # Distances from output sample point to input sample points
    # weights = h(dists)
    weights = h(np.expand_dims(u, axis=1) - indices.astype(np.float64) - 1) # -1 because indexing from 0. Ensure indices are float for subtraction.

    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1),
                        out=np.zeros_like(weights), where=np.expand_dims(np.sum(weights, axis=1), axis=1)!=0) # Avoid division by zero

    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)] # Reflection padding

    # Store only non-zero columns
    # Ensure ind2store is applied correctly, especially if weights can be all zero for some columns
    # (though normalization should handle sum=0 cases if they lead to NaN/inf)
    # If a column in weights is all zero, it means that input pixel does not contribute to any output pixel.
    # If a row sum is zero (before normalization), it means output pixel gets no contribution (e.g. outside range of kernel from all inputs).
    # The normalization step handles the row sum.

    # Original ind2store logic:
    # ind2store = np.nonzero(np.any(weights, axis=0))
    # weights = weights[:, ind2store[0]] # if ind2store is a tuple of arrays
    # indices = indices[:, ind2store[0]]

    # A slightly more robust way if `np.any` could result in an empty `ind2store`
    # though with bicubic interpolation this is unlikely unless out_length is very small.
    valid_cols_mask = np.any(weights, axis=0)
    if np.any(valid_cols_mask): # Check if there is at least one column with non-zero weights
        weights = weights[:, valid_cols_mask]
        indices = indices[:, valid_cols_mask]
    else:
        # This case should ideally not happen for typical image resizing.
        # If it does, it means no input pixels contribute to any output pixels,
        # which implies the weights matrix would be all zeros.
        # The shape of weights and indices might become problematic if all columns are removed.
        # For now, assume it won't happen with valid inputs.
        # Or, ensure weights and indices retain their P dimension, e.g. by not sub-selecting if all are zero.
        # However, the original code structure implies sub-selection.
        pass # Stick to original behavior of potentially reducing columns

    return weights, indices

def imresizemex(inimg, weights, indices, dim):
    in_shape = inimg.shape
    w_shape = weights.shape
    out_shape = list(in_shape)
    out_shape[dim] = w_shape[0]
    outimg = np.zeros(out_shape, dtype=np.float64) # Ensure output is float for calculations

    if dim == 0: # Interpolating along columns (height)
        for i_w in range(w_shape[0]): # For each output row
            # w_row are the weights for this output row, ind_row are the input row indices
            w_row = weights[i_w, :]
            ind_row = indices[i_w, :]
            # Pixel slice from input image: selected rows, all columns
            # For each column in the input image, apply interpolation
            for i_img_col in range(in_shape[1]): # Iterate over columns of the image
                im_slice = inimg[ind_row, i_img_col].astype(np.float64)
                outimg[i_w, i_img_col] = np.sum(im_slice * w_row) # Element-wise then sum
    elif dim == 1: # Interpolating along rows (width)
        for i_w in range(w_shape[0]): # For each output col
            w_row = weights[i_w, :]
            ind_row = indices[i_w, :]
            # For each row in the input image, apply interpolation
            for i_img_row in range(in_shape[0]): # Iterate over rows of the image
                im_slice = inimg[i_img_row, ind_row].astype(np.float64)
                outimg[i_img_row, i_w] = np.sum(im_slice * w_row)

    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        # For float images, clipping might also be desired depending on expected range (e.g. 0-1)
        return outimg

def imresizevec(inimg, weights, indices, dim):
    outimg = None # Initialize outimg
    # Ensure inimg is float for calculations
    # If inimg is uint8, its values are 0-255. Calculations should be float.
    inimg_float = inimg.astype(np.float64)

    w_shape = weights.shape # (out_dim_len, num_contrib_pixels)

    # Debug prints
    # print(f"Debug imresizevec: dim={dim}, inimg_float.shape={inimg_float.shape}, weights.shape={weights.shape}, indices.shape={indices.shape}")

    if dim == 0: # Interpolating along columns (dimension 0)
        # Output shape: (out_dim_len, in_shape[1], in_shape[2]...)
        # weights: (w_shape[0], w_shape[1]) -> needs to be (w_shape[0], w_shape[1], 1, ...) for broadcasting
        # indices: (w_shape[0], w_shape[1])

        gathered_pixels = inimg_float[indices] # Shape: (L_out, N_coeffs, D1, D2, ...)
        reshaped_weights = weights.reshape(w_shape[0], w_shape[1], *((1,)*(inimg_float.ndim - 1)))

        # print(f"  dim=0: gathered_pixels.shape={gathered_pixels.shape}, reshaped_weights.shape={reshaped_weights.shape}")

        outimg = np.sum(gathered_pixels * reshaped_weights, axis=1)
        # print(f"  dim=0: outimg.shape after sum: {outimg.shape if outimg is not None else 'None'}")


    elif dim == 1:
        # Interpolating along rows (dimension 1)
        img_perm = np.transpose(inimg_float, np.roll(np.arange(inimg_float.ndim), -dim))
        # Now img_perm has shape (L_in, D0, D2, ...), we interpolate its axis 0

        gathered_pixels = img_perm[indices] # Shape: (L_out, N_coeffs, D0, D2, ...)
        reshaped_weights = weights.reshape(w_shape[0], w_shape[1], *((1,)*(img_perm.ndim - 1)))

        # print(f"  dim=1: img_perm.shape={img_perm.shape}, gathered_pixels.shape={gathered_pixels.shape}, reshaped_weights.shape={reshaped_weights.shape}")

        interpolated_permuted = np.sum(gathered_pixels * reshaped_weights, axis=1) # Shape (L_out, D0, D2, ...)
        # print(f"  dim=1: interpolated_permuted.shape: {interpolated_permuted.shape if interpolated_permuted is not None else 'None'}")

        inv_perm_indices = np.argsort(np.roll(np.arange(inimg_float.ndim), -dim))
        outimg = np.transpose(interpolated_permuted, inv_perm_indices)
        # print(f"  dim=1: outimg.shape after transpose: {outimg.shape if outimg is not None else 'None'}")

    # print(f"  outimg.shape before clip/return: {outimg.shape if outimg is not None else 'None'}")

    if outimg is None:
        # This case should not be reached if dim is always 0 or 1 and calculations are correct.
        # Raise an error or handle appropriately if it can be.
        # For now, if it's None, the clip below will fail, which is what we're seeing.
        print(f"Error: outimg is None before clipping in imresizevec (dim={dim})")


    if inimg.dtype == np.uint8:
        # The following line will fail if outimg is None
        outimg_clipped = np.clip(outimg, 0, 255)
        return np.around(outimg_clipped).astype(np.uint8)
    else:
        return outimg


def resizeAlongDim(A, dim, weights, indices, mode="vec"):
    if mode == "org": # Original, likely refers to the loop-based version
        out = imresizemex(A, weights, indices, dim)
    else: # Default to vectorized version
        out = imresizevec(A, weights, indices, dim)
    return out

def imresize(I, scalar_scale=None, method='bicubic', output_shape=None, mode="vec"):
    if method == 'bicubic':
        kernel = cubic
        kernel_width = 4.0
    elif method == 'bicubic_hw_friendly': # New method
        kernel = hardware_friendly_cubic
        kernel_width = 4.0
    elif method == 'bilinear':
        kernel = triangle
        kernel_width = 2.0 # Bilinear kernel width is 2
    else:
        raise ValueError('unidentified kernel method supplied: {}'.format(method))

    # Fill scale and output_size
    if scalar_scale is not None and output_shape is not None:
        raise ValueError('either scalar_scale OR output_shape should be defined')
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        if scalar_scale <= 0:
            raise ValueError('scalar_scale must be positive')
        scale = [scalar_scale, scalar_scale]
        output_size = deriveSizeFromScale(I.shape, scale)
    elif output_shape is not None:
        if any(s <= 0 for s in output_shape):
            raise ValueError('output_shape dimensions must be positive')
        scale = deriveScaleFromSize(I.shape, output_shape)
        output_size = list(output_shape) # Ensure it's a list
    else:
        raise ValueError('either scalar_scale OR output_shape should be defined')

    # If input image has more than 2 dimensions (e.g., color image)
    # The interpolation is done separately for each channel.
    # The original code handles 2D and adds/squeezes a 3rd dim.
    # A more general approach is to process the first two dims and iterate over others if needed,
    # or ensure vectorization handles higher dims correctly.

    # The current `contributions` and `resizeAlongDim` are designed for 2D images,
    # or applying to 2D slices of a 3D+ image.
    # The `imresizevec` was modified to attempt handling >2D images.

    # Determine order of operations (which dimension to process first)
    # Original code sorts by scale factor. This is a common optimization.
    # scale_np = np.array(scale)
    # order = np.argsort(scale_np) # Process dimension with smaller scale factor first
    order = [0, 1] # Simpler: process rows, then columns (or vice-versa consistently)
                   # The provided code does not show argsort affecting which dim is 0 or 1,
                   # but rather the loop `for k in range(2): dim = order[k]`
                   # Let's stick to a fixed order [0, 1] for simplicity unless specific issues arise.
                   # The original `order = np.argsort(scale_np)` is fine too. Let's restore it.
    scale_np = np.array(scale)
    order = np.argsort(scale_np)


    weights_all_dims = []
    indices_all_dims = []

    # Calculate weights and indices for each dimension that is being scaled
    # This assumes I.shape has at least 2 dimensions.
    # If I is 1D, this loop structure might need adjustment. (Current problem is 2D images)
    for k_dim_idx in range(len(I.shape[:2])): # Iterate over first two dimensions
        dim_size_in = I.shape[k_dim_idx]
        dim_size_out = output_size[k_dim_idx]
        dim_scale = scale[k_dim_idx]

        w, ind = contributions(dim_size_in, dim_size_out, dim_scale, kernel, kernel_width)
        weights_all_dims.append(w)
        indices_all_dims.append(ind)

    B = np.copy(I).astype(np.float64) # Work with float64 for precision

    # Handle potential flag2D for single channel images passed as 2D
    # This logic is from the original user code.
    # It ensures that even if a 2D grayscale image is passed, it's temporarily treated as 3D
    # with one channel for consistent processing by resizeAlongDim if that function expects 3D.
    # However, resizeAlongDim (and its sub-functions) should ideally handle N-D inputs
    # where the first two dimensions are spatial.

    # The original code did:
    # if B.ndim == 2: B = np.expand_dims(B, axis=2); flag2D = True
    # This suggests resizeAlongDim might be tailored for 3D (H,W,C) inputs.
    # My imresizevec attempts to be more general. Let's see.

    # For simplicity and focusing on the kernel, let's assume B is processed as is.
    # If B is (H,W), dim 0 is H, dim 1 is W.
    # If B is (H,W,C), dim 0 is H, dim 1 is W. `resizeAlongDim` must handle the C channels correctly.
    # The `imresizevec` I wrote should handle this by permuting axes.

    for k_pass in range(2): # Two passes for separable interpolation
        dim_to_process = order[k_pass] # e.g., first rows (dim 0), then columns (dim 1) or vice-versa

        current_weights = weights_all_dims[dim_to_process]
        current_indices = indices_all_dims[dim_to_process]

        B = resizeAlongDim(B, dim_to_process, current_weights, current_indices, mode)

    # Final type conversion (if original was uint8)
    if I.dtype == np.uint8:
        B = np.clip(B, 0, 255)
        B = np.around(B).astype(np.uint8)
    # else, B is already float64, could be converted to original float type if needed.

    return B


def convertDouble2Byte(I):
    B = np.clip(I, 0.0, 1.0) # Assumes input I is in range [0,1] for float
    B = 255*B
    return np.around(B).astype(np.uint8)

# Example usage (for testing purposes, will be in a separate test file)
if __name__ == '__main__':
    # Create a dummy image
    img_2d = np.random.rand(32, 32) * 255
    img_2d_uint8 = img_2d.astype(np.uint8)

    img_3d = np.random.rand(32, 32, 3) * 255
    img_3d_uint8 = img_3d.astype(np.uint8)

    output_shape_2d = (64, 64)
    output_shape_3d = (64, 64, 3) # Output shape for imresize should only be 2D part

    print("Testing 2D uint8 image...")
    resized_2d_orig = imresize(img_2d_uint8, output_shape=output_shape_2d, method='bicubic', mode='vec')
    print("Original bicubic (vec) output shape:", resized_2d_orig.shape)
    resized_2d_hw = imresize(img_2d_uint8, output_shape=output_shape_2d, method='bicubic_hw_friendly', mode='vec')
    print("HW-friendly bicubic (vec) output shape:", resized_2d_hw.shape)

    # Check if outputs are close (should be identical given the math is equivalent for floats)
    if resized_2d_orig.shape == resized_2d_hw.shape:
        diff_2d = np.sum(np.abs(resized_2d_orig.astype(np.float64) - resized_2d_hw.astype(np.float64)))
        print("Difference between original and HW-friendly (2D uint8):", diff_2d) # Should be near 0

    print("\nTesting 3D uint8 image...")
    # output_shape for imresize should define only spatial dimensions
    resized_3d_orig = imresize(img_3d_uint8, output_shape=output_shape_2d, method='bicubic', mode='vec')
    print("Original bicubic (vec) output shape:", resized_3d_orig.shape)
    resized_3d_hw = imresize(img_3d_uint8, output_shape=output_shape_2d, method='bicubic_hw_friendly', mode='vec')
    print("HW-friendly bicubic (vec) output shape:", resized_3d_hw.shape)

    if resized_3d_orig.shape == resized_3d_hw.shape:
        diff_3d = np.sum(np.abs(resized_3d_orig.astype(np.float64) - resized_3d_hw.astype(np.float64)))
        print("Difference between original and HW-friendly (3D uint8):", diff_3d) # Should be near 0

    print("\nTesting 2D float image (0-1 range)...")
    img_2d_float = np.random.rand(32, 32)
    resized_2d_float_orig = imresize(img_2d_float, output_shape=output_shape_2d, method='bicubic', mode='vec')
    resized_2d_float_hw = imresize(img_2d_float, output_shape=output_shape_2d, method='bicubic_hw_friendly', mode='vec')
    if resized_2d_float_orig.shape == resized_2d_float_hw.shape:
        diff_2d_float = np.sum(np.abs(resized_2d_float_orig - resized_2d_float_hw))
        print("Difference between original and HW-friendly (2D float):", diff_2d_float) # Should be very close to 0

    # Test case: scale down
    print("\nTesting scale down (2D uint8)...")
    img_large_uint8 = np.random.randint(0, 256, (128,128), dtype=np.uint8)
    resized_down_hw = imresize(img_large_uint8, output_shape=(64,64), method='bicubic_hw_friendly', mode='vec')
    print("HW-friendly bicubic (vec) scale down output shape:", resized_down_hw.shape)
    resized_down_orig = imresize(img_large_uint8, output_shape=(64,64), method='bicubic', mode='vec')
    if resized_down_hw.shape == resized_down_orig.shape:
        diff_down = np.sum(np.abs(resized_down_hw.astype(np.float64) - resized_down_orig.astype(np.float64)))
        print("Difference (scale down):", diff_down)

    # Test bilinear as well
    print("\nTesting bilinear (2D uint8)...")
    resized_bilinear = imresize(img_2d_uint8, output_shape=output_shape_2d, method='bilinear', mode='vec')
    print("Bilinear output shape:", resized_bilinear.shape)

    # Test with scalar scale
    print("\nTesting with scalar_scale (2D uint8)...")
    resized_scalar_hw = imresize(img_2d_uint8, scalar_scale=2.0, method='bicubic_hw_friendly', mode='vec')
    print("HW-friendly bicubic (scalar_scale=2.0) output shape:", resized_scalar_hw.shape)
    assert resized_scalar_hw.shape == output_shape_2d # Should be (64,64) for 32x32 input
    resized_scalar_orig = imresize(img_2d_uint8, scalar_scale=2.0, method='bicubic', mode='vec')
    if resized_scalar_hw.shape == resized_scalar_orig.shape:
        diff_scalar = np.sum(np.abs(resized_scalar_hw.astype(np.float64) - resized_scalar_orig.astype(np.float64)))
        print("Difference (scalar_scale):", diff_scalar)

    # Test with 'org' mode (loop-based)
    print("\nTesting 2D uint8 image with 'org' mode...")
    resized_2d_orig_loop = imresize(img_2d_uint8, output_shape=output_shape_2d, method='bicubic', mode='org')
    print("Original bicubic (org) output shape:", resized_2d_orig_loop.shape)
    resized_2d_hw_loop = imresize(img_2d_uint8, output_shape=output_shape_2d, method='bicubic_hw_friendly', mode='org')
    print("HW-friendly bicubic (org) output shape:", resized_2d_hw_loop.shape)
    if resized_2d_orig_loop.shape == resized_2d_hw_loop.shape:
        diff_2d_loop = np.sum(np.abs(resized_2d_orig_loop.astype(np.float64) - resized_2d_hw_loop.astype(np.float64)))
        print("Difference between original and HW-friendly (2D uint8, org mode):", diff_2d_loop)

    # Test the imresizemex directly for a small case
    print("\nDirectly testing imresizemex (internal logic check)")
    test_input_mex = np.array([[10,20,30],[40,50,60],[70,80,90]], dtype=np.uint8)
    # Say we want to expand dim 0 from 3 to 4 pixels using some dummy weights/indices
    # Output pixel 0: 0.5*row0 + 0.5*row1
    # Output pixel 1: 0.5*row1 + 0.5*row2
    # Output pixel 2: 0.5*row0 + 0.25*row1 + 0.25*row2
    # Output pixel 3: 1.0*row2
    dummy_weights_mex = np.array([[0.5,0.5,0],[0,0.5,0.5],[0.25,0.25,0.5],[0,0,1.0]])
    dummy_indices_mex = np.array([[0,1,0],[1,2,0],[0,1,2],[2,0,0]]) # last two indices in each row are dummies if weight is 0

    # Limit indices to actual size of input for this test
    dummy_indices_mex = np.array([[0,1],[1,2],[0,1,2],[2]]) # This structure is not exactly how contributions generates it
                                                          # contributions ensures indices and weights have same num_coeffs

    # Let's use a contributions-like output for dummy:
    # Target output size 4 for a dim of size 3. Scale = 4/3.
    # For simplicity, let's make weights for 2 contributions per output pixel
    dummy_weights_mex_2 = np.array([[0.75, 0.25], [0.5, 0.5], [0.25, 0.75], [0.1, 0.9]]) # (4, 2)
    dummy_indices_mex_2 = np.array([[0,1],[0,1],[1,2],[1,2]], dtype=np.int32) # (4,2)

    out_mex_dim0 = imresizemex(test_input_mex, dummy_weights_mex_2, dummy_indices_mex_2, 0)
    print("imresizemex output for dim 0:\n", out_mex_dim0)
    # Expected for first col, first output pixel: 0.75*10 + 0.25*40 = 7.5 + 10 = 17.5 -> 18
    # Expected for first col, second output pixel: 0.5*10 + 0.5*40 = 5 + 20 = 25
    # Expected for first col, third output pixel: 0.25*40 + 0.75*70 = 10 + 52.5 = 62.5 -> 63
    # Expected for first col, fourth output pixel: 0.1*40 + 0.9*70 = 4 + 63 = 67

    out_mex_dim1 = imresizemex(test_input_mex, dummy_weights_mex_2, dummy_indices_mex_2, 1)
    print("imresizemex output for dim 1:\n", out_mex_dim1)
    # Expected for first row, first output pixel: 0.75*10 + 0.25*20 = 7.5 + 5 = 12.5 -> 13

    print("Done with local tests.")
