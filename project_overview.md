# Project Overview: Hardware-Friendly Bicubic Interpolation - Algorithm Comparison

## 1. Introduction

### 1.1. Project Background
Image scaling via bicubic interpolation offers a good balance between output quality and computational complexity compared to simpler methods. However, for hardware implementations (FPGAs/ASICs), the "standard" bicubic algorithm still presents challenges due to its reliance on floating-point arithmetic and general multiplications. This project explores optimizing bicubic interpolation for hardware, focusing on reducing resource usage while quantifying image quality impacts through Python simulation.

### 1.2. Project Goal & Specifications
The primary goals are:
1.  To implement a traditional bicubic interpolation algorithm as a baseline.
2.  To develop a hardware-friendly bicubic algorithm using floating-point arithmetic, where kernel coefficients are modified for easier hardware translation (e.g., into shifts and adds).
3.  To simulate this hardware-friendly algorithm using fixed-point arithmetic to understand the impact of finite precision.
4.  To compare these three versions in terms of image quality (PSNR) and, secondarily, Python execution time.

**Specifications:**
*   **Input Image:** Primarily 256x256 pixels (grayscale and color).
*   **Output Image:** Primarily 512x512 pixels (2x upscale).
*   **Optimization Target:** Reduce complexity of applying kernel coefficients in hardware.
*   **Quality Metric:** Peak Signal-to-Noise Ratio (PSNR).
*   **Simulation Environment:** Python with NumPy and scikit-image.

## 2. Algorithm Versions and Implementation Details

### 2.1. `traditional_bicubic.py` - Baseline Implementation
*   **Purpose:** Serves as the reference for correctness and quality.
*   **Algorithm:** Implements the standard bicubic interpolation using Keys' cubic kernel with `a = -0.5`. Coefficients are `1.5, -2.5, -0.5`, etc.
*   **Kernel `cubic(x)`:** Direct floating-point implementation of the piecewise cubic polynomial.
*   **`imresize()` function:** Main interface.
    *   `mode='org'`: Loop-based implementation (`imresizemex`), generally more robust for various image dimensions if logic is sound.
    *   `mode='vec'`: Original vectorized version (`imresizevec`) provided by the user, which has known limitations in handling N-D data and its internal reshape logic. This is kept for fidelity but may not be reliable for all comparisons.
*   **Key Challenge:** Direct hardware implementation of floating-point coefficient multipliers is resource-intensive.

### 2.2. `optimized_bicubic_float.py` - Hardware-Friendly (Float)
*   **Purpose:** To show that the bicubic kernel can be mathematically reformulated for hardware without quality loss in an ideal (floating-point) scenario.
*   **Algorithm:** Uses the same bicubic theory but re-expresses kernel calculations.
*   **Kernel `hardware_friendly_cubic(x_float)`:**
    *   Coefficients are represented as exact fractions. For example, `1.5*val` is calculated as `(3*val)/2`.
    *   All arithmetic is still standard Python floating-point.
    *   This demonstrates the *mathematical transformation* step before considering fixed-point effects.
*   **`imresize_optimized_float()` function:**
    *   Uses the `hardware_friendly_cubic` kernel.
    *   Employs a robust N-D vectorized implementation (`imresizevec_optimized`) for `mode='vec'`, and a corrected N-D loop-based version (`imresizemex_optimized`) for `mode='org'`.
*   **Expected Hardware Benefit (Conceptual):** Operations like `(3*val)/2` can translate to `((val << 1) + val) >> 1` in hardware, replacing a general multiplier with shifters and an adder.

### 2.3. `optimized_bicubic_fixed_point.py` - Hardware-Friendly (Fixed-Point Simulation)
*   **Purpose:** To simulate the `optimized_bicubic_float.py` algorithm under fixed-point arithmetic constraints, providing an estimate of image quality degradation due to finite precision.
*   **Fixed-Point Simulation:**
    *   **Parameters:**
        *   `FP_W_Kernel=16, FP_F_Kernel=8`: For kernel calculations (distances, intermediate weights). Range: approx -128.0 to +127.996.
        *   `FP_W_Pixel=24, FP_F_Pixel=8`: For pixel data representation during interpolation and for final scaled weights. Range allows for accumulation.
    *   **Helper Functions:** `float_to_fixed`, `fixed_to_float`, `fixed_add`, `fixed_subtract`, `fixed_multiply`, `saturate`. These simulate fixed-point behavior including saturation for overflows. `fixed_multiply` handles scaling based on fractional bits of inputs and output.
*   **Kernel `hardware_friendly_cubic_fixed_point(x_float)`:**
    *   Input distance `x_float` is converted to fixed-point (`x_fixed`).
    *   All internal calculations (`absx2_fixed`, `absx3_fixed`, application of coefficients like `3`, `5`, `2.0`) use the fixed-point helper functions.
    *   The final division by 2 is simulated by `int(np.round(num / 2.0))`, approximating a right shift with rounding.
    *   Output is a fixed-point weight (scaled integer with `FP_F_Kernel` fractional bits).
*   **`contributions_fixed_point()`:**
    *   The fixed-point kernel's output is temporarily converted to float for numerically stable normalization of weights.
    *   Normalized float weights are then converted back to fixed-point (with `FP_F_Pixel` fractional bits) for application to pixel data. This avoids implementing a full fixed-point division for normalization.
*   **`imresize_fixed_point()` (via `imresizevec_fixed_point`):**
    *   Input `uint8` image data is converted to fixed-point (`FP_F_Pixel` fractional bits, unsigned).
    *   Pixel data (fixed-point) is multiplied by weights (fixed-point) using `fixed_multiply`.
    *   Products are summed (accumulation). The accumulator maintains `FP_F_Pixel` fractional accuracy.
    *   Final accumulated fixed-point values are converted back to float, then clipped to [0,255] and rounded to `uint8`.
    *   Currently, only `mode='vec'` is implemented, using element-wise loops for fixed-point operations within the vectorized structure for clarity.
*   **Key Insight:** This version reveals the PSNR impact of choosing specific bit-widths and the effects of quantization/overflow.

## 3. Simulation Setup and Comparison (`compare_all_versions.py`)

*   **Test Images:** Generates 256x256 grayscale and color test images with varied patterns.
*   **Upscaling:** All versions upscale images to 512x512.
*   **Execution:**
    *   Calls the main `imresize` function from each of the three files.
    *   Tests both `'org'` and `'vec'` modes where available and appropriate.
*   **Metrics:**
    *   **PSNR:**
        *   `optimized_float` vs. `traditional` (Ideally using `traditional`'s `org` mode as the most stable baseline).
        *   `optimized_fixed_point` vs. `traditional` (or `optimized_float`).
        *   `optimized_fixed_point` vs. `optimized_float` (to isolate fixed-point effects).
    *   **Execution Time:** Python execution time for each method (provides a rough performance indication in the simulation environment).

## 4. Expected Simulation Summary & Analysis

*   **Optimized Float vs. Traditional:**
    *   **PSNR:** Expected to be extremely high (approaching infinity if `traditional_bicubic.py` (org mode) output is used as reference), as the `optimized_bicubic_float.py` is mathematically equivalent. Any minor difference would be due to floating-point operation ordering.
    *   **Hardware Implication:** Confirms that the coefficient reformulation itself doesn't degrade quality, validating the first step of optimization.

*   **Optimized Fixed-Point vs. Optimized Float (or Traditional):**
    *   **PSNR:** Expected to be lower than the pure float comparison. The magnitude of the drop will depend on the chosen `W` and `F` values for `FP_F_Kernel` and `FP_F_Pixel`.
        *   Too few fractional bits (`F`) can lead to significant quantization errors in weights, distances, and pixel values.
        *   Too few integer bits (`W-F`) can lead to overflow/saturation, clipping details.
    *   **Analysis:** The goal is to find fixed-point parameters that yield a "good enough" PSNR (e.g., >30-35 dB is often acceptable, but application-specific) while allowing for minimal bit-widths in hardware.
    *   The PSNR difference between `optimized_fixed_point` and `optimized_float` will most directly show the impact of the simulated fixed-point arithmetic.

*   **Execution Times:**
    *   While Python execution times don't directly map to hardware speed, significant differences in the Python versions might hint at algorithmic complexity differences that *could* translate to hardware, but this is a very indirect measure. The primary focus for hardware is resource reduction from the algorithm's structure.

*   **Overall Hardware Implication:** The project aims to show a clear path from a standard algorithm to one that is structurally simpler for hardware (fewer/no general multipliers for coefficients). The fixed-point simulation then provides a crucial estimate of how much "real-world" image quality might be affected when precision is necessarily limited in hardware. This informs trade-offs in actual hardware design.

This structured comparison will provide valuable insights into the feasibility and impact of these hardware-friendly optimizations for bicubic interpolation.
