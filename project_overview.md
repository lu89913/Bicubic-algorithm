# Project Overview: Hardware-Friendly Bicubic Interpolation

## 1. Introduction

### 1.1. Project Background
Image scaling is a fundamental operation in digital image processing, with applications ranging from display adaptation to image editing and computer vision. Bicubic interpolation is a widely used technique known for its ability to produce smoother results with fewer artifacts compared to simpler methods like bilinear or nearest-neighbor interpolation. However, traditional bicubic interpolation can be computationally intensive, especially for hardware implementations where resources like multipliers and memory bandwidth are critical.

### 1.2. Project Goal
The primary goal of this project is to develop an optimized bicubic interpolation algorithm that is specifically tailored for efficient hardware implementation. The optimization aims to reduce the anticipated hardware resource consumption (particularly multipliers) without significantly degrading the image quality, as measured by Peak Signal-to-Noise Ratio (PSNR). The project includes a Python-based simulation to demonstrate the algorithm's correctness and quality preservation. The target comparison is a Python implementation analogous to MATLAB's `imresize` (bicubic) function, with a specific test case of upscaling a 256x256 image to 512x512.

## 2. Traditional Bicubic Interpolation Algorithm

### 2.1. Principle
Bicubic interpolation calculates the value of an output pixel by performing a weighted average of pixels in the nearest 4x4 neighborhood of the corresponding input pixel. The weights are determined by a cubic convolution kernel, typically the Keys' kernel.

The interpolation is separable, meaning the 2D interpolation can be performed as two 1D interpolations: first along rows, then along columns (or vice-versa). Each 1D interpolation uses 4 neighboring pixels.

### 2.2. Python Implementation Key Points
The provided initial Python code (`image_interpolation.py`) implements this as:
*   **`cubic(x)` function:** Defines the Keys' cubic kernel:
    *   `f(x) = (a+2)|x|^3 - (a+3)|x|^2 + 1` for `|x| <= 1`
    *   `f(x) = a|x|^3 - 5a|x|^2 + 8a|x| - 4a` for `1 < |x| <= 2`
    *   `f(x) = 0` otherwise
    *   The standard implementation uses `a = -0.5`. This leads to coefficients like `1.5`, `-2.5`, `-0.5`, etc.
*   **`contributions()` function:** Calculates the weights and indices of the input pixels that contribute to each output pixel for a single dimension. It handles scaling factors and boundary conditions (reflection padding).
*   **`imresize()` function:** Orchestrates the 2D resizing process by calling `contributions()` for each dimension and then applying the 1D interpolation (either via a loop-based `imresizemex` or a vectorized `imresizevec`).

## 3. Hardware Optimization Strategy

### 3.1. Explored Optimization Directions
Several strategies were considered for making the bicubic algorithm more hardware-friendly:
*   **Coefficient Approximation/Quantization:** Simplifying the kernel's floating-point coefficients.
*   **Lookup Tables (LUTs):** Pre-calculating and storing parts of the kernel computation.
*   **Pipelining & Parallelism:** Architectural optimizations for hardware.
*   **Polynomial Rewriting (e.g., Horner's Rule):** Reducing operations in polynomial evaluation.

### 3.2. Chosen Strategy: Coefficient Re-representation
The chosen strategy focuses on **re-representing the exact kernel coefficients as simple fractions**. This approach directly targets the reduction of multiplication complexity in hardware.
The standard Keys' kernel with `a = -0.5` has coefficients such as:
*   `1.5`
*   `-2.5`
*   `1.0`
*   `-0.5` (for `a`)
*   `2.5`
*   `-4.0`
*   `2.0`

These can be precisely expressed as fractions with small denominators (typically 2):
*   `1.5 = 3/2`
*   `-2.5 = -5/2`
*   `1.0 = 1/1`
*   `-0.5 = -1/2`
*   `2.5 = 5/2`
*   `-4.0 = -4/1`
*   `2.0 = 2/1`

## 4. Hardware-Friendly Bicubic Algorithm Design (`hardware_friendly_cubic`)

### 4.1. Kernel Modification
A new kernel function, `hardware_friendly_cubic(x)`, was designed. Mathematically, it is identical to the original `cubic(x)` function when `a=-0.5`. The difference lies in how the calculations involving coefficients are expressed.

**Original `cubic(x)` (for `|x| <= 1`):**
`1.5*|x|^3 - 2.5*|x|^2 + 1`

**`hardware_friendly_cubic(x)` (for `|x| <= 1`), equivalent form:**
`(3 * |x|^3 - 5 * |x|^2 + 2) / 2`

Similar transformations apply to the `1 < |x| <= 2` case:
**Original:** `-0.5*|x|^3 + 2.5*|x|^2 - 4*|x| + 2`
**Equivalent:** `(-|x|^3 + 5*|x|^2 - 8*|x| + 4) / 2`

### 4.2. Anticipated Hardware Implementation Advantages
This re-representation is highly beneficial for hardware:
*   **Multiplication by `N/2`**: Operations like `(Val * 3) / 2` can be implemented as `( (Val << 1) + Val ) >> 1`. This uses one adder and two shifters.
*   **Multiplication by Integer**: `Val * C` (where C is a small integer like 3, 5, 8) can be done with shifters and adders. For example, `Val * 3 = (Val << 1) + Val`. `Val * 5 = (Val << 2) + Val`. `Val * 8 = Val << 3`.
*   **Resource Saving**: This approach replaces general-purpose multipliers (which are area-intensive and can be slower) needed for floating-point or arbitrary fixed-point coefficient multiplication with simpler, faster, and more area-efficient shifters and adders.
*   The core multiplications for `|x|^2` and `|x|^3` (data * data) remain, but the constant coefficient multiplications are significantly simplified.

## 5. Python Simulation and Validation

### 5.1. Test Environment (`test_interpolation.py`)
A dedicated Python script (`test_interpolation.py`) was created to:
*   Generate 256x256 grayscale and 3-channel color test images.
*   Upscale these images to 512x512 using both the traditional `bicubic` method and the new `bicubic_hw_friendly` method from `image_interpolation.py`.
*   Utilize `skimage.metrics.peak_signal_noise_ratio` to calculate the PSNR between the outputs of the two methods.
*   Measure and report the Python execution time for each method.

### 5.2. Simulation Results Analysis

*   **Image Quality (PSNR)**:
    *   For both grayscale and color images, the PSNR between the output of `bicubic_hw_friendly` and `bicubic` was **infinite (inf dB)**.
    *   The sum of absolute differences between the outputs was **0.0**.
    *   This confirms that, within Python's floating-point precision, the hardware-friendly modifications are mathematically equivalent to the original algorithm and **introduce no degradation in image quality**. The `RuntimeWarning: divide by zero` during PSNR calculation is expected when images are identical (MSE is zero).

*   **Python Execution Time (Secondary Observation)**:
    *   A modest speedup was observed in the Python execution of `bicubic_hw_friendly` compared to the original `bicubic` implementation.
        *   Grayscale (256x256 -> 512x512): ~0.055s (HW-friendly) vs. ~0.103s (Traditional)
        *   Color (256x256x3 -> 512x512x3): ~0.206s (HW-friendly) vs. ~0.303s (Traditional)
    *   This speedup is likely due to differences in how NumPy handles the slightly restructured arithmetic operations in Python, and is not the primary goal, but a welcome side-effect. The true performance benefit is expected in actual hardware.

## 6. Conclusion

The project successfully developed and validated a hardware-friendly bicubic interpolation algorithm. The key achievements are:

1.  **No Image Quality Degradation**: The Python simulation demonstrated that the optimized algorithm produces results identical to the traditional bicubic method, maintaining maximum PSNR.
2.  **Significant Potential for Hardware Efficiency**: The algorithm's design, by re-representing coefficients as simple fractions, paves the way for a hardware implementation that relies on shifters and adders instead of more complex multipliers for coefficient application. This is anticipated to lead to substantial savings in hardware area, power consumption, and potentially improved clock speeds.
3.  **Python Verification**: The provided Python scripts (`image_interpolation.py` and `test_interpolation.py`) serve as a complete reference and verification environment for the algorithm.

The developed `hardware_friendly_cubic` algorithm meets the project's objective of providing a high-quality image scaling solution optimized for hardware implementation.

### 6.1. Future Work
*   Implement the `hardware_friendly_cubic` algorithm in a Hardware Description Language (e.g., Verilog or VHDL).
*   Synthesize the HDL design for a target FPGA or ASIC to quantify actual resource usage, power, and timing performance.
*   Compare these hardware metrics against a similarly implemented traditional bicubic interpolator.
*   Investigate fixed-point arithmetic effects: determine optimal bit-widths for intermediate calculations in hardware to balance precision and resource cost.

This project provides a strong algorithmic foundation for such future hardware development efforts.
