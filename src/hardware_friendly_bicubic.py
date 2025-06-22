import numpy as np

# --- Fixed-point parameters ---
F_BITS = 8  # Number of fractional bits
F_SCALE = 1 << F_BITS # 256

# Parameters for cubic kernel with a = -0.5
# All are scaled by F_SCALE
A_FIXED = int(-0.5 * F_SCALE)  # -128
A_PLUS_2_FIXED = int(1.5 * F_SCALE) # 384 (a+2)
A_PLUS_3_FIXED = int(2.5 * F_SCALE) # 640 (a+3)

# Coefficients for the second segment (1 < x <= 2) when a = -0.5
# a * x^3 - 5a * x^2 + 8a * x - 4a
# -0.5 * x^3 + 2.5 * x^2 - 4 * x + 2
COEFF_X3_S2 = A_FIXED                 # -0.5 -> -128
COEFF_X2_S2 = int(2.5 * F_SCALE)      #  2.5 ->  640
COEFF_X1_S2 = int(-4.0 * F_SCALE)     # -4.0 -> -1024
COEFF_C_S2  = int(2.0 * F_SCALE)      #  2.0 ->  512

def fixed_round_shift(value, shift_bits):
    """Rounds to nearest, then shifts right. Equivalent to round(value / (2**shift_bits))."""
    if shift_bits == 0:
        return value
    rounding_val = 1 << (shift_bits - 1)
    return (value + rounding_val) >> shift_bits

def cubic_kernel_fixed_point(x_fixed_q2_f): # Input x is Q2.F_BITS (abs value)
    """
    Computes the bicubic kernel weights using fixed-point arithmetic.
    x_fixed_q2_f: Absolute distance from the sample point, in Q2.F_BITS format.
                  (e.g., if F_BITS=8, 0.0 is 0, 1.0 is 256, 2.0 is 512)
    Returns kernel weight in QsX.F_BITS format (internal calculations might need more bits).
    Let's aim for output as Qs4.F_BITS (signed, 4 integer, F_BITS fractional)
    Intermediate precision for x^2, x^3:
    x (Q2.F) -> x^2 (Q4.2F) -> x^3 (Q6.3F)
    We need to shift them back to QX.F before multiplying with coefficients (which are QsY.F)
    Or, multiply first then shift the larger product.
    Example: (coeff_qX.F * x_cubed_q6.3F) -> Q(X+6).(4F) -> Shift by 3F to get Q(X+6).F
    """
    # Ensure x_fixed is within expected range for kernel calculation [0, 2*F_SCALE)
    # x_fixed represents values from 0 up to (but not including) 2.0
    
    # x_fixed is Q2.F (e.g. range 0 to 511 for F_BITS=8)
    # x_sq_fixed will be Q4.2F
    # x_cub_fixed will be Q6.3F
    x_sq_fixed = (x_fixed_q2_f * x_fixed_q2_f) 
    x_cub_fixed = (x_sq_fixed * x_fixed_q2_f)

    # Branch 1: 0 <= x < 1 (i.e., 0 <= x_fixed < F_SCALE)
    if x_fixed_q2_f < F_SCALE:
        # Formula: (a+2)x^3 - (a+3)x^2 + 1
        # (A_PLUS_2_FIXED * x^3) - (A_PLUS_3_FIXED * x^2) + 1*F_SCALE*F_SCALE*F_SCALE (approx)
        # Branch 1: 0 <= x < 1 (i.e., 0 <= x_fixed < F_SCALE)
        # Formula: 1.5 * x^3 - 2.5 * x^2 + 1
        
        # Scale x_cub_fixed (Q6.3F) and x_sq_fixed (Q4.2F) to QX.F for operations
        x_cub_qX_f = fixed_round_shift(x_cub_fixed, 2 * F_BITS) # Now Q6.F (approx)
        x_sq_qX_f  = fixed_round_shift(x_sq_fixed, F_BITS)    # Now Q4.F (approx)

        # Term 1: 1.5 * x^3 = x^3 + (x^3 >> 1)
        term1 = x_cub_qX_f + fixed_round_shift(x_cub_qX_f, 1)
        
        # Term 2: 2.5 * x^2 = (x^2 << 1) + (x^2 >> 1)
        term2 = (x_sq_qX_f << 1) + fixed_round_shift(x_sq_qX_f, 1)
        
        # Constant 1, represented as QX.F (e.g. 1.0 * F_SCALE)
        one_fixed_qX_f = 1 * F_SCALE
        
        result_fixed = term1 - term2 + one_fixed_qX_f

    # Branch 2: 1 <= x < 2 (i.e., F_SCALE <= x_fixed < 2*F_SCALE)
    elif x_fixed_q2_f < 2 * F_SCALE:
        # Formula for a=-0.5: -0.5*x^3 + 2.5*x^2 - 4*x + 2
        
        x_cub_qX_f = fixed_round_shift(x_cub_fixed, 2 * F_BITS) # Q6.F
        x_sq_qX_f  = fixed_round_shift(x_sq_fixed, F_BITS)    # Q4.F
        x_qX_f     = x_fixed_q2_f # Q2.F (already appropriately scaled for this context if F_BITS is the common scale)

        # Term x3: -0.5 * x^3 = -(x^3 >> 1)
        t_x3 = -fixed_round_shift(x_cub_qX_f, 1)
        
        # Term x2: 2.5 * x^2 = (x^2 << 1) + (x^2 >> 1)
        t_x2 = (x_sq_qX_f << 1) + fixed_round_shift(x_sq_qX_f, 1)
        
        # Term x1: -4 * x = -(x << 2)
        t_x1 = -(x_qX_f << 2)
        
        # Term c: +2
        t_c = 2 * F_SCALE
        
        result_fixed = t_x3 + t_x2 + t_x1 + t_c
    else:
        # x >= 2
        result_fixed = 0

    return int(result_fixed) # Ensure integer result for fixed point representation

def bicubic_interpolation_pixel_fixed_point(p_uint8, tx_fixed_q0_f, ty_fixed_q0_f):
    """
    Performs bicubic interpolation for a single pixel using fixed-point arithmetic.
    p_uint8 (np.ndarray): A 4x4 numpy array of neighboring pixel values (uint8).
    tx_fixed_q0_f (int): Fractional x-coordinate in Q0.F_BITS format (0 to F_SCALE-1).
    ty_fixed_q0_f (int): Fractional y-coordinate in Q0.F_BITS format (0 to F_SCALE-1).
    Returns:
        int: The interpolated pixel value (0-255).
    """

    # Calculate x values for kernel: 1+tx, tx, 1-tx, 2-tx
    # tx_fixed is Q0.F. F_SCALE is 1.0 in Q0.F
    # (1+tx) -> F_SCALE + tx_fixed (Q1.F, needs to be handled as Q2.F for kernel input)
    # (tx)   -> tx_fixed (Q0.F)
    # (1-tx) -> F_SCALE - tx_fixed (Q1.F if positive, Q0.F if tx is large)
    # (2-tx) -> 2*F_SCALE - tx_fixed (Q2.F)
    
    # Kernel expects absolute values in Q2.F format.
    # Max value of tx_fixed is F_SCALE-1.
    # Distances for wx (all positive):
    # d0 = 1+tx  -> F_SCALE + tx_fixed_q0_f  (range F_SCALE to 2*F_SCALE-1, Q1.F, fits Q2.F)
    # d1 = tx    -> tx_fixed_q0_f            (range 0 to F_SCALE-1, Q0.F, fits Q2.F)
    # d2 = 1-tx  -> F_SCALE - tx_fixed_q0_f  (range 1 to F_SCALE, Q1.F if tx!=0, Q0.F if tx=0, fits Q2.F)
    # d3 = 2-tx  -> 2*F_SCALE - tx_fixed_q0_f(range F_SCALE+1 to 2*F_SCALE, Q2.F)
    
    # Note: kernel function takes abs distance. Here, these are already positive distances.
    wx_fixed = np.array([
        cubic_kernel_fixed_point(F_SCALE + tx_fixed_q0_f), # 1+tx
        cubic_kernel_fixed_point(tx_fixed_q0_f),           # tx
        cubic_kernel_fixed_point(F_SCALE - tx_fixed_q0_f), # 1-tx
        cubic_kernel_fixed_point(2 * F_SCALE - tx_fixed_q0_f)  # 2-tx (this was error in logic, should be abs(x-(idx)) )
    ])
    
    # Corrected distances for kernel (kernel input is |coord_diff|):
    # For wx, the 4 relevant grid points are at x = -1, 0, 1, 2
    # Interpolation point is at x_int + tx.
    # Distances are | (x_int + tx) - x_grid |
    # wx[0] corresponds to grid point at -1. Distance = |tx - (-1)| = |tx + 1| = 1+tx
    # wx[1] corresponds to grid point at  0. Distance = |tx - 0|   = |tx|     = tx
    # wx[2] corresponds to grid point at  1. Distance = |tx - 1|   = 1-tx (since 0<=tx<1)
    # wx[3] corresponds to grid point at  2. Distance = |tx - 2|   = 2-tx (since 0<=tx<1)
    
    # These map to the x_fixed_q2_f inputs:
    # For wx[0]: kernel_arg = F_SCALE + tx_fixed_q0_f (for distance 1+tx)
    # For wx[1]: kernel_arg = tx_fixed_q0_f           (for distance tx)
    # For wx[2]: kernel_arg = F_SCALE - tx_fixed_q0_f (for distance 1-tx)
    # For wx[3]: kernel_arg = 2*F_SCALE - tx_fixed_q0_f if using 2-tx, or F_SCALE + (F_SCALE - tx_fixed_q0_f)
    # Let's re-verify kernel arguments for wx:
    # Kernel argument is distance. For indices i = -1, 0, 1, 2 for the 4x4 patch.
    # wx[0] = w(1+tx)
    # wx[1] = w(tx)
    # wx[2] = w(1-tx)
    # wx[3] = w(2-tx)
    # This seems correct.

    wy_fixed = np.array([
        cubic_kernel_fixed_point(F_SCALE + ty_fixed_q0_f), # 1+ty
        cubic_kernel_fixed_point(ty_fixed_q0_f),           # ty
        cubic_kernel_fixed_point(F_SCALE - ty_fixed_q0_f), # 1-ty
        cubic_kernel_fixed_point(2 * F_SCALE - ty_fixed_q0_f)  # 2-ty
    ])

    # Weights wx_fixed, wy_fixed are QsX.F_BITS (e.g. Qs4.8)
    # Pixel values p_uint8 are uint8 (Q8.0 effectively)
    
    # Interpolation: result = wy^T * p * wx
    # Step 1: temp_coeffs[j] = sum(p[j,i] * wx_fixed[i]) for i=0..3
    # p[j,i] (uint8) * wx_fixed[i] (Qs4.8) -> product is Qs12.8 (signed, 8+4=12 int, 8 frac)
    # Sum of 4 such products: Qs14.8 (add 2 bits for integer part for sum of 4)
    
    temp_coeffs = np.zeros(4, dtype=np.int64) # Use int64 for intermediate sums
    for j in range(4):
        current_sum = 0
        for i in range(4):
            # p_uint8 elements are 0-255. wx_fixed elements can be negative.
            # Product: p_val * w_val (implicit scaling by F_SCALE due to w_val)
            current_sum += p_uint8[j, i] * wx_fixed[i]
        temp_coeffs[j] = current_sum
        
    # temp_coeffs are Qs14.8 (approx)

    # Step 2: result_sum = sum(temp_coeffs[j] * wy_fixed[j]) for j=0..3
    # temp_coeffs[j] (Qs14.8) * wy_fixed[j] (Qs4.8) -> product is Qs(14+4).(8+8) = Qs18.16
    # Sum of 4 such products: Qs20.16
    
    final_sum_q_2f = 0
    for j in range(4):
        final_sum_q_2f += temp_coeffs[j] * wy_fixed[j]
        
    # final_sum_q_2f is now QsX.2*F_BITS (e.g., Qs20.16)
    # We need to shift it back to QsX.0 (i.e., an integer pixel value)
    # This involves shifting right by 2 * F_BITS and rounding.
    
    interpolated_value_scaled = fixed_round_shift(final_sum_q_2f, 2 * F_BITS)
    
    # Clip to valid pixel range (0-255)
    if interpolated_value_scaled < 0:
        interpolated_value_scaled = 0
    elif interpolated_value_scaled > 255:
        interpolated_value_scaled = 255
        
    return int(interpolated_value_scaled)

def bicubic_resize_fixed_point(image_uint8, scale_factor_x, scale_factor_y):
    """
    Resizes an image using fixed-point bicubic interpolation.
    Args:
        image_uint8 (np.ndarray): Input image (grayscale, 2D numpy array, dtype=np.uint8).
        scale_factor_x (float): Scaling factor for x-axis.
        scale_factor_y (float): Scaling factor for y-axis.
    Returns:
        np.ndarray: Resized image (dtype=np.uint8).
    """
    if image_uint8.ndim != 2 or image_uint8.dtype != np.uint8:
        raise ValueError("Input image must be a 2D grayscale image of dtype uint8.")

    in_height, in_width = image_uint8.shape
    out_height = int(np.ceil(in_height * scale_factor_y))
    out_width = int(np.ceil(in_width * scale_factor_x))
    
    output_image = np.zeros((out_height, out_width), dtype=np.uint8)

    padded_image = np.pad(image_uint8, pad_width=2, mode='reflect')

    for j_out in range(out_height): # y_out
        for i_out in range(out_width): # x_out
            x_in_float = i_out / scale_factor_x
            y_in_float = j_out / scale_factor_y

            x_int = int(np.floor(x_in_float))
            y_int = int(np.floor(y_in_float))

            # Fractional parts as floats (0 to <1)
            tx_float = x_in_float - x_int
            ty_float = y_in_float - y_int
            
            # Convert fractional parts to Q0.F_BITS fixed-point
            tx_fixed = int(round(tx_float * F_SCALE))
            ty_fixed = int(round(ty_float * F_SCALE))
            # Clamp to ensure they are less than F_SCALE (e.g. if tx_float was almost 1.0)
            tx_fixed = min(tx_fixed, F_SCALE -1)
            ty_fixed = min(ty_fixed, F_SCALE -1)

            p_uint8 = padded_image[y_int + 1 : y_int + 1 + 4,
                                   x_int + 1 : x_int + 1 + 4]
            
            interpolated_val_uint8 = bicubic_interpolation_pixel_fixed_point(p_uint8, tx_fixed, ty_fixed)
            output_image[j_out, i_out] = interpolated_val_uint8
                
    return output_image

if __name__ == '__main__':
    # Small test case
    original_image = np.array([
        [10, 20, 30, 40],
        [50, 60, 70, 80],
        [90, 100, 110, 120],
        [130, 140, 150, 160]
    ], dtype=np.uint8)

    print("Original Image (Fixed-Point Test):\n", original_image)
    scale_x = 1.5
    scale_y = 1.5
    
    resized_image_fixed = bicubic_resize_fixed_point(original_image, scale_x, scale_y)
    print(f"\nResized Image Fixed-Point (scale {scale_x}x{scale_y}):\n", resized_image_fixed)
    print("Output shape:", resized_image_fixed.shape)

    # Compare with traditional float for this small case
    from traditional_bicubic import bicubic_resize as bicubic_resize_float
    resized_image_float = bicubic_resize_float(original_image, scale_x, scale_y)
    print(f"\nResized Image Float (for comparison):\n", resized_image_float)
    
    mse = np.mean((resized_image_fixed.astype(np.float64) - resized_image_float.astype(np.float64))**2)
    print(f"\nMSE between fixed-point and float implementations (small test): {mse:.4f}")

    # Try with the gradient image
    try:
        from PIL import Image
        grad_img_pil = Image.open("../images/gradient.png").convert('L')
        grad_img_np = np.array(grad_img_pil, dtype=np.uint8)
        
        print("\n--- Gradient Image Test ---")
        resized_grad_fixed = bicubic_resize_fixed_point(grad_img_np, scale_x, scale_y)
        resized_grad_float = bicubic_resize_float(grad_img_np, scale_x, scale_y)
        
        mse_grad = np.mean((resized_grad_fixed.astype(np.float64) - resized_grad_float.astype(np.float64))**2)
        print(f"Resized Gradient Fixed shape: {resized_grad_fixed.shape}")
        print(f"Resized Gradient Float shape: {resized_grad_float.shape}")
        print(f"MSE between fixed-point and float for gradient image: {mse_grad:.4f}")
        if mse_grad == 0: psnr_grad = float('inf')
        else: psnr_grad = 20 * np.log10(255.0 / np.sqrt(mse_grad))
        print(f"PSNR between fixed-point and float for gradient image: {psnr_grad:.2f} dB")

    except ImportError:
        print("\nPillow not installed, skipping gradient image test in main.")
    except FileNotFoundError:
        print("\nGradient image not found, skipping gradient image test in main.")
