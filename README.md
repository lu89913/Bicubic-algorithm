# Hardware-Friendly Bicubic Interpolation: Algorithm Comparison

## 1. Project Overview

This project develops and compares different Python implementations of the bicubic interpolation algorithm, focusing on optimizing for hardware efficiency while analyzing the impact on image quality (PSNR). It includes:
1.  A **traditional bicubic interpolation** algorithm, based on common Python translations of MATLAB's `imresize` behavior.
2.  An **optimized bicubic algorithm using floating-point arithmetic** (`optimized_bicubic_float.py`), where kernel coefficients are represented as precise fractions to simplify hardware implementation (e.g., using shifters and adders instead of general multipliers).
3.  An **optimized bicubic algorithm simulating fixed-point arithmetic** (`optimized_bicubic_fixed_point.py`), which further models the quantization and overflow effects that would occur in a hardware implementation.
4.  A comparison script (`compare_all_versions.py`) to evaluate these three versions.

The primary goal is to demonstrate that hardware-friendly optimizations can be achieved, and to simulate the expected image quality trade-offs when moving towards a fixed-point hardware design. The main test case is upscaling a 256x256 image to 512x512.

## 2. Core Algorithm Files

*   **`traditional_bicubic.py`**:
    *   Contains the baseline bicubic interpolation algorithm.
    *   Intended to mirror the behavior of standard bicubic implementations.
    *   Includes `imresize()` which can be run in `'org'` (loop-based) or `'vec'` (vectorized, with known limitations in this original version) mode.

*   **`optimized_bicubic_float.py`**:
    *   Implements the hardware-friendly bicubic kernel (`hardware_friendly_cubic`) using floating-point numbers.
    *   The kernel coefficients are mathematically equivalent to the traditional kernel but are expressed as simple fractions (e.g., 1.5 is handled as 3/2).
    *   Provides `imresize_optimized_float()` with robust `'org'` and `'vec'` modes.

*   **`optimized_bicubic_fixed_point.py`**:
    *   Simulates the hardware-friendly bicubic algorithm using fixed-point arithmetic.
    *   Defines fixed-point parameters (total width `W`, fractional width `F`).
    *   Includes helper functions for fixed-point operations (addition, multiplication, saturation).
    *   The `hardware_friendly_cubic_fixed_point` kernel and `imresize_fixed_point` function operate using these simulated fixed-point numbers.
    *   This version helps estimate potential quality degradation due to finite precision in hardware. (Currently, only `vec` mode is implemented).

## 3. Comparison Script

*   **`compare_all_versions.py`**:
    *   Imports the `imresize` functions from the three algorithm files.
    *   Generates test images (grayscale and color).
    *   Runs image upscaling using all three implementations.
    *   Calculates and reports:
        *   PSNR between "Optimized Float" and "Traditional".
        *   PSNR between "Optimized Fixed-Point" and "Traditional" (or "Optimized Float").
        *   Execution times for each version.

## 4. Dependencies

*   Python 3.x
*   NumPy
*   scikit-image (for PSNR calculation)

Install dependencies using pip:
```bash
pip install numpy scikit-image
```

## 5. How to Run the Comparison

1.  Ensure Python and the required dependencies are installed.
2.  Navigate to the project directory in your terminal.
3.  Run the comparison script:
    ```bash
    python compare_all_versions.py
    ```

## 6. Expected Output & Results Summary

The `compare_all_versions.py` script will output:
*   Execution status and timings for each algorithm version on test images.
*   A summary table of PSNR values and timings.

**Expected PSNR observations:**
*   **Optimized Float vs. Traditional**: PSNR should be very high (ideally `inf dB` if the `traditional_bicubic.py`'s `org` mode is used as a stable reference), indicating no quality loss from the coefficient re-representation in a floating-point environment.
*   **Optimized Fixed-Point vs. Traditional/Optimized Float**: PSNR will likely be lower than the float comparison, showing the impact of fixed-point quantization. The magnitude of this drop will depend on the chosen fixed-point bit-widths (`W`, `F`). A good result would be a high PSNR (e.g., >30-35 dB, application-dependent) indicating acceptable quality for many hardware applications.

This setup allows for a clear understanding of the algorithmic trade-offs from a baseline, through hardware-aware float optimization, to a simulated fixed-point implementation.

## 7. Project Overview Document

For a more detailed explanation of the project's background, algorithm design choices, fixed-point simulation details, and an in-depth analysis of results, please refer to `project_overview.md`.
