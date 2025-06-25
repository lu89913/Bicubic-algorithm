from __future__ import print_function
import numpy as np
from math import ceil, floor
from PIL import Image
import math
import time # For timing

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

def cubic(x): # This is the bicubic kernel
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    # Standard cubic convolution kernel with a = -0.5 (implicit)
    # f = (1.5*absx3 - 2.5*absx2 + 1) for |x| <= 1
    # f = (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) for 1 < |x| <= 2
    # This is equivalent to:
    # if |x| <= 1: (a+2)|x|^3 - (a+3)|x|^2 + 1
    # if 1 < |x| <= 2: a|x|^3 - 5a|x|^2 + 8a|x| - 4a
    # where a = -0.5
    # For |x| <= 1: (-0.5+2)|x|^3 - (-0.5+3)|x|^2 + 1 = 1.5|x|^3 - 2.5|x|^2 + 1 (Matches)
    # For 1 < |x| <= 2: -0.5|x|^3 - 5(-0.5)|x|^2 + 8(-0.5)|x| - 4(-0.5)
    #                 = -0.5|x|^3 + 2.5|x|^2 - 4|x| + 2 (Matches)
    f = np.multiply(1.5*absx3 - 2.5*absx2 + 1, absx <= 1) + \
        np.multiply(-0.5*absx3 + 2.5*absx2 - 4*absx + 2, (1 < absx) & (absx <= 2))
    return f

def contributions(in_length, out_length, scale, kernel, k_width):
    """
    Calculate the contributions of input pixels to an output pixel.
    This involves determining which input pixels are relevant and their weights.
    Returns 0-based indices.
    """
    if scale < 1: # Downsampling
        h = lambda x: scale * kernel(scale * x) # Adjust kernel for downsampling
        kernel_width = 1.0 * k_width / scale
    else: # Upsampling
        h = kernel
        kernel_width = k_width

    x_coords = np.arange(1, out_length + 1).astype(np.float64) # 1-based output pixel coordinates
    u = x_coords / scale + 0.5 * (1 - 1 / scale) # Mapped to 1-based input coordinate system

    left = np.floor(u - kernel_width / 2) # Leftmost relevant input pixel (1-based index)

    P = int(ceil(kernel_width)) + 2 # Number of kernel taps

    # `ind_1based` will be 1-based indices of contributing input pixels
    # Each row corresponds to an output pixel. Columns are the input pixel indices.
    # Example: if left = [10.0] (1-based), P=4, np.arange(P)-1 = [-1,0,1,2]
    # ind_1based for that output pixel = [10-1, 10+0, 10+1, 10+2] = [9,10,11,12] (1-based)
    ind_1based = np.expand_dims(left, axis=1) + (np.arange(P) - 1)

    # Calculate weights: h(distance from output_pixel_center_in_input_coords to input_pixel_center_coords)
    # u_expanded: (out_length, 1) - 1-based centers of output pixels in input space
    # ind_1based_float: (out_length, P) - 1-based centers of relevant input pixels
    # The original MATLAB code: weights = h(bsxfun(@minus, u', indices)); where u and indices are 1-based.
    # The term `... - indices.astype(np.float64) -1)` in the original Python code was confusing.
    # If ind_1based are 1-based centers, then `u_expanded - ind_1based_float` is the correct distance.
    # Let's use this direct interpretation.
    weights = h(np.expand_dims(u, axis=1) - ind_1based.astype(np.float64))

    sum_weights = np.sum(weights, axis=1, keepdims=True)
    weights = np.divide(weights, sum_weights, out=np.zeros_like(weights), where=sum_weights!=0)

    # Convert 1-based indices to 0-based for Python array indexing
    indices_0based = ind_1based.astype(np.int32) - 1

    # Boundary handling using mirroring (reflect padding)
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices_0based_mirrored = aux[np.mod(indices_0based, aux.size)]

    valid_cols_mask = np.any(weights, axis=0)
    if np.any(valid_cols_mask):
        weights = weights[:, valid_cols_mask]
        indices_0based_mirrored = indices_0based_mirrored[:, valid_cols_mask]

    return weights, indices_0based_mirrored


def imresizemex(inimg, weights, indices_0based, dim):
    """
    Interpolation using loops. indices_0based must be 0-based.
    """
    inimg_c = inimg.astype(np.float64)
    out_shape = list(inimg_c.shape)
    out_shape[dim] = weights.shape[0]
    outimg = np.zeros(out_shape, dtype=np.float64)

    if dim == 0:
        for i_col in range(inimg_c.shape[1]):
            for i_out_row in range(weights.shape[0]):
                w_row_coeffs = weights[i_out_row, :]
                ind_row_input_pixels = indices_0based[i_out_row, :]
                pixel_slice = inimg_c[ind_row_input_pixels, i_col]
                outimg[i_out_row, i_col] = np.sum(pixel_slice * w_row_coeffs)
    elif dim == 1:
        for i_row in range(inimg_c.shape[0]):
            for i_out_col in range(weights.shape[0]):
                w_col_coeffs = weights[i_out_col, :]
                ind_col_input_pixels = indices_0based[i_out_col, :]
                pixel_slice = inimg_c[i_row, ind_col_input_pixels]
                outimg[i_row, i_out_col] = np.sum(pixel_slice * w_col_coeffs)

    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg

def imresizevec(inimg, weights, indices_0based, dim):
    """
    Interpolation using vectorized operations. indices_0based must be 0-based.
    """
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

    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg

def resizeAlongDim(A, dim, weights, indices_0based, mode="vec"):
    if mode == "org":
        out = imresizemex(A, weights, indices_0based, dim)
    else:
        out = imresizevec(A, weights, indices_0based, dim)
    return out

def imresize(I, scalar_scale=None, method='bicubic', output_shape=None, mode="vec"):
    if method == 'bicubic':
        kernel = cubic
        kernel_width = 4.0
    elif method == 'bilinear':
        kernel = triangle
        kernel_width = 2.0
    else:
        raise ValueError(f'Unidentified kernel method supplied: {method}')

    if scalar_scale is not None and output_shape is not None:
        raise ValueError('Either scalar_scale OR output_shape should be defined')
    if scalar_scale is not None:
        try:
            scalar_scale = float(scalar_scale)
        except ValueError:
            raise ValueError('scalar_scale must be a number')
        if scalar_scale <= 0:
            raise ValueError('scalar_scale must be positive')
        scale = [scalar_scale, scalar_scale]
        output_size = deriveSizeFromScale(I.shape[:2], scale)
    elif output_shape is not None:
        if not (isinstance(output_shape, (list, tuple)) and len(output_shape) == 2):
            raise ValueError('output_shape must be a list or tuple of 2 elements')
        if any(s <= 0 for s in output_shape):
            raise ValueError('output_shape dimensions must be positive')
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
        w, ind_0based = contributions(dim_size_in, dim_size_out, dim_scale, kernel, kernel_width)
        weights_all_dims.append(w)
        indices_all_dims.append(ind_0based)

    B = np.copy(I)

    for k_pass in range(2):
        dim_to_process = order[k_pass]
        current_weights = weights_all_dims[dim_to_process]
        current_indices_0based = indices_all_dims[dim_to_process]
        B = resizeAlongDim(B, dim_to_process, current_weights, current_indices_0based, mode)

    return B

def calculate_psnr_pil(img1_path, img2_path, max_pixel_value=255):
    try:
        img1 = Image.open(img1_path).convert('L')
        img2 = Image.open(img2_path).convert('L')
    except FileNotFoundError:
        print(f"Error: One or both image files not found for PSNR: {img1_path}, {img2_path}")
        return None

    img1_array = np.array(img1, dtype=np.float64)
    img2_array = np.array(img2, dtype=np.float64)

    if img1_array.shape != img2_array.shape:
        print(f"Warning: Image dimensions mismatch for PSNR. {img1_path}: {img1_array.shape}, {img2_path}: {img2_array.shape}")
        img2 = img2.resize(img1.size, Image.BICUBIC)
        img2_array = np.array(img2, dtype=np.float64)
        if img1_array.shape != img2_array.shape:
             print(f"Error: Still mismatched after resize. Cannot calculate PSNR.")
             return None

    return calculate_psnr_arrays(img1_array, img2_array, max_pixel_value)

def calculate_psnr_arrays(arr1, arr2, max_pixel_value=255):
    if arr1.shape != arr2.shape:
        print(f"Error: Array dimensions must match for PSNR. Got {arr1.shape} and {arr2.shape}")
        return None
    mse = np.mean((arr1.astype(np.float64) - arr2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse))
    return psnr

if __name__ == '__main__':
    print("Running tests for traditional_bicubic.py (GitHub version with corrections)...")

    # Assuming script is run from project root (/app)
    golden_img_path = "lena_golden_512.png"
    input_img_path = "lena_downscaled_256.png"
    output_dir = "." # Output images to the current directory (/app)

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

        print("\nTesting 'org' mode (loop-based)...")
        resized_org = imresize(input_array, output_shape=target_shape, method='bicubic', mode='org')
        print(f"Resized 'org' mode shape: {resized_org.shape}")
        psnr_org = calculate_psnr_arrays(golden_array, resized_org)
        if psnr_org is not None: print(f"PSNR (org vs golden): {psnr_org:.4f} dB")
        if resized_org.shape == target_shape:
            Image.fromarray(resized_org.astype(np.uint8)).save(f"{output_dir}/lena_traditional_bicubic_org_512.png")
            print(f"Saved 'org' mode result to {output_dir}/lena_traditional_bicubic_org_512.png")

        print("\nTesting 'vec' mode (vectorized)...")
        start_time_vec = time.time()
        resized_vec = imresize(input_array, output_shape=target_shape, method='bicubic', mode='vec')
        end_time_vec = time.time()
        print(f"Time taken for traditional 'vec' mode: {end_time_vec - start_time_vec:.4f} seconds")
        print(f"Resized 'vec' mode shape: {resized_vec.shape}")
        psnr_vec = calculate_psnr_arrays(golden_array, resized_vec)
        if psnr_vec is not None: print(f"PSNR (vec vs golden): {psnr_vec:.4f} dB")
        if resized_vec.shape == target_shape:
            Image.fromarray(resized_vec.astype(np.uint8)).save(f"{output_dir}/lena_traditional_bicubic_vec_512.png")
            print(f"Saved 'vec' mode result to {output_dir}/lena_traditional_bicubic_vec_512.png")

        pillow_resized_img = input_img_pil.resize(golden_img_pil.size, Image.BICUBIC)
        pillow_resized_array = np.array(pillow_resized_img)
        psnr_pillow = calculate_psnr_arrays(golden_array, pillow_resized_array)
        if psnr_pillow is not None: print(f"\nPSNR (Pillow BICUBIC vs golden): {psnr_pillow:.4f} dB")
        pillow_resized_img.save(f"{output_dir}/lena_pillow_bicubic_512.png")
        print(f"Saved Pillow BICUBIC result to {output_dir}/lena_pillow_bicubic_512.png")

    except FileNotFoundError as e:
        print(f"Error: Could not find Lena image files. Expected at {golden_img_path} and {input_img_path}.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"An error occurred during image processing tests: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Original small array tests ---")
    img_gray_small = np.array([[10,20,30,40],[50,60,70,80],[90,100,110,120],[130,140,150,160]], dtype=np.uint8)
    print(f"Original grayscale shape (small): {img_gray_small.shape}")
    try:
        resized_gray_org_small = imresize(img_gray_small, output_shape=(6,6), method='bicubic', mode='org')
        print(f"Resized grayscale (org mode, small) shape: {resized_gray_org_small.shape}")
    except Exception as e: print(f"Error in small org test: {e}"); import traceback; traceback.print_exc()
    try:
        resized_gray_vec_small = imresize(img_gray_small, output_shape=(6,6), method='bicubic', mode='vec')
        print(f"Resized grayscale (vec mode, small) shape: {resized_gray_vec_small.shape}")
    except Exception as e: print(f"Error in small vec test: {e}"); import traceback; traceback.print_exc()
    print("Traditional bicubic script tests completed.")
