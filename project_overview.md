# 專案總覽：優化 Bicubic 內插演算法及其定點化

## 1. 專案背景與目標

本專案旨在對 Bicubic 圖像內插演算法進行優化和定點化轉換。主要目標包括：

1.  **開發優化浮點版本 (`optimized_bicubic_float.py`)**:
    *   在保持或提升圖像品質 (以 PSNR 衡量) 的前提下，大幅簡化 Bicubic 演算法核心的運算過程。簡化的目標是將複雜的數學運算（如浮點冪運算）替換為更適合硬體或低階語言實現的形式，例如查表。
    *   此版本不追求在 Python/NumPy 環境下超越高度優化的原生 NumPy 運算速度，而是側重於運算結構的改變。
2.  **開發定點版本 (`optimized_bicubic_fixed_point.py`)**:
    *   基於上述優化的浮點版本，將其轉換為定點數實現。
    *   確保定點版本能夠在可接受的精度損失範圍內（PSNR 與浮點版本相近）工作。
3.  **驗證與比較**:
    *   使用標準的 Lena 圖像進行測試。
    *   比較不同版本演算法的 PSNR 和執行時間（在 Python 環境下）。

## 2. 圖像預處理

*   **測試圖像**: 選用影像處理領域常用的 Lena 圖像。
*   **Golden 圖像**: 一張 512x512 分辨率的 Lena 圖像 (`lena_golden_512.png`) 作為原始參考。
*   **輸入圖像**: Golden 圖像通過 Bicubic 降採樣得到一張 256x256 分辨率的圖像 (`lena_downscaled_256.png`)，作為各內插演算法的輸入。
*   **預處理腳本**: `Bicubic-algorithm/preprocess_image.py` 用於自動下載 Lena 圖像並進行上述處理。

## 3. 傳統 Bicubic 演算法分析 (`traditional_bicubic.py`)

*   **代碼來源**: 本專案中使用的 `traditional_bicubic.py` 是基於 GitHub 用戶 `lu89913` 的 [Bicubic-algorithm](https://github.com/lu89913/Bicubic-algorithm) 倉庫中的實現。我們對其進行了小幅調整以適應本專案的測試框架（如路徑、PSNR 計算集成等）。
*   **核心原理**:
    *   **Bicubic 核函數**: 採用標準的立方卷積核函數 (`cubic(x)`)，該函數通常隱含參數 `a = -0.5`。其數學表達式為：
        *   `f(x) = (a+2)|x|^3 - (a+3)|x|^2 + 1`  for `|x| <= 1`
        *   `f(x) = a|x|^3 - 5a|x|^2 + 8a|x| - 4a` for `1 < |x| <= 2`
        *   `f(x) = 0` otherwise
    *   **可分離性**: Bicubic 內插是可分離的，可以先對圖像進行水平方向的一維 Bicubic 內插，然後對結果再進行垂直方向的一維 Bicubic 內插（反之亦然）。
*   **關鍵函數**:
    *   `cubic(x)`: 計算上述核函數值。在 `lu89913` 的版本中，此函數已使用 NumPy 進行向量化操作。
    *   `contributions(...)`: 此函數非常關鍵，它為輸出圖像的每個像素（在一維處理中）計算所有相關的輸入像素的索引 (`indices`) 和它們對應的權重 (`weights`)。權重是通過 `cubic(distance)` 計算得到的。此函數處理了邊界條件（鏡像填充）和權重歸一化。
    *   `imresizevec(...)`: 使用 NumPy 向量化操作執行插值。它調用 `contributions` 計算權重和索引，然後通過矩陣/向量操作完成加權求和。
    *   `imresizemex(...)`: 一個循環版本的插值實現，主要用於對比或理解。在本專案中，我們主要關注 `imresizevec` 的性能。
*   **基準性能**:
    *   在我們的測試環境中，使用 `lena_downscaled_256.png` 放大到 512x512，`traditional_bicubic.py` 的 `vec` (向量化) 模式得到的 PSNR 約為 **34.1076 dB**，執行時間約為 **0.0488 秒**。 Pillow 庫的 `Image.BICUBIC` 給出的 PSNR 約為 34.1102 dB，兩者非常接近。

## 4. 優化浮點版本開發 (`optimized_bicubic_float.py`)

*   **優化目標達成策略**:
    *   **運算簡化**: 主要的簡化思路是將 `cubic(x)` 核函數的實時計算替換為 **查找表 (LUT)**。這樣可以避免重複的浮點乘法和冪運算。
    *   **PSNR 保持**: LUT 的精度（子像素級別劃分）需要足夠高，以確保查表引入的誤差在可接受範圍內。
*   **LUT 設計與實現**:
    *   `CUBIC_LUT_FLOAT`: 一個一維 NumPy 數組，存儲預先計算好的 `cubic(x)` 值。
    *   `SUBPIXEL_LEVELS = 128`: 將一個像素內的子像素位置劃分為 128 個級別。LUT 的大小由 `KERNEL_RADIUS` (Bicubic 為 2.0) 和 `SUBPIXEL_LEVELS` 決定。LUT 存儲 `x` 在 `[0, KERNEL_RADIUS]` 範圍內的值，利用對稱性處理負距離。
    *   `precompute_bicubic_lut()`: 負責生成此浮點 LUT。
*   **`contributions_lut(...)` 函數**:
    *   這是 `traditional_bicubic.py` 中 `contributions` 函數的 LUT 版本。
    *   距離計算 (`distances`) 仍然是浮點數。
    *   通過 `abs_distances * SUBPIXEL_LEVELS` 並四捨五入、裁剪得到 LUT 索引。
    *   直接從 `CUBIC_LUT_FLOAT` 數組中獲取核函數值（權重因子）。此過程完全向量化，避免了 Python 級別的循環或 `np.vectorize` 的低效。
    *   後續的權重縮放（針對降採樣）、邊界外權重置零、權重歸一化等步驟與原版 `contributions` 邏輯類似，均為向量化操作。
*   **`imresize_optimized_float(...)`**: 主函數結構與 `traditional_bicubic.py` 中的 `imresize` 類似，但調用 `contributions_lut`。
*   **結果與分析**:
    *   **PSNR**: 34.1076 dB。與傳統向量化版本完全一致，表明 `SUBPIXEL_LEVELS = 128` 的 LUT 精度足夠。
    *   **執行時間**: 約 0.1271 秒。
    *   **性能討論**: 此 LUT 版本比 `traditional_bicubic.py` (0.0488 秒) 要慢。原因在於 `traditional_bicubic.py` 中的 `cubic()` 函數本身已經是高效的 NumPy 向量化實現，其底層是優化的 C 代碼。而 LUT 版本雖然避免了 Python 層的 `cubic()` 函數調用，但增加了額外的 NumPy 操作（如計算索引、查表本身、裁剪索引等）。在 Python 環境中，這些額外操作的開銷超過了 `cubic()` 函數的原始開銷。
    *   **「簡化」的意義**: 儘管此 Python 實現未帶來速度優勢，但「查表」確實是一種運算結構的簡化。它將一個可能複雜的數學函數替換為內存訪問和簡單的算術運算，這對於沒有高效浮點單元或需要極低功耗的硬體實現，以及需要手動實現定點化的低階語言環境來說，是一種非常重要的優化策略。

## 5. 定點版本開發 (`optimized_bicubic_fixed_point.py`)

*   **目標**: 將 `optimized_bicubic_float.py` 的 LUT 方法轉換為定點實現，並驗證其精度。
*   **定點策略**:
    *   **權重表示**: 選擇 `LUT_FRAC_BITS = 10`。這意味著浮點權重 `w` 將被轉換為 `round(w * 2^10)` 的整數。Bicubic 權重範圍大致在 `[-0.21, 1.0]`，轉換後約為 `[-215, 1024]`，適合用 `int16` 存儲 (腳本中使用 `int32` 以簡化類型轉換，但實際需要的位寬較小)。
    *   **像素值**: 輸入圖像像素為 `uint8` (0-255)。在計算時提升至 `int32` 以避免與定點權重相乘時溢出。
    *   **中間累加**: `pixel (int32) * fixed_weight (int32)` 的乘積，以及後續的加權和，都在 `int32` 下進行。對於4個點的一維插值，`255 * 1024` 約為 `2.6 * 10^5`，4個這樣的和約為 `10^6`，遠在 `int32`範圍內。
*   **關鍵函數修改**:
    *   `precompute_bicubic_lut()`: 保持不變，它生成 `CUBIC_LUT_FLOAT` 作為後續計算的基礎。
    *   `contributions_fixed_point_weights(...)`:
        1.  與 `contributions_lut` 類似，首先使用 `CUBIC_LUT_FLOAT` 計算出歸一化前的浮點核函數值 (`float_kernel_values`)。
        2.  對這些 `float_kernel_values` 進行歸一化，使其每行（對應一個輸出點的所有貢獻權重）之和為1，得到 `normalized_float_weights`。
        3.  將 `normalized_float_weights` 乘以 `2^LUT_FRAC_BITS` 並四捨五入，轉換為 `fixed_point_weights` (整數)。這些即為最終用於插值的定點權重。
    *   `imresizevec_fixed_point(...)`:
        1.  接收 `fixed_point_weights`。
        2.  計算 `weighted_sum = sum(pixel_as_int32 * fixed_weight)`。
        3.  **縮放和四捨五入**: 通過 `scaled_result = floor((weighted_sum / (2^LUT_FRAC_BITS)) + 0.5)` 將 `weighted_sum` 還原到像素值範圍。這等效於右移 `LUT_FRAC_BITS` 位並進行正確的四捨五入。
        4.  將結果裁剪到 `[0, 255]` 並轉換回 `uint8`。
    *   `imresize_optimized_fixed_point(...)`: 主調用函數，結構類似浮點版本，但調用定點處理函數並傳遞 `LUT_FRAC_BITS`。
*   **結果與分析**:
    *   **PSNR**: 34.1079 dB。與浮點版本 (34.1076 dB) 非常接近，表明 `LUT_FRAC_BITS = 10` 的精度是足夠的，定點化引入的誤差極小。
    *   **執行時間**: 約 0.1000 秒。
    *   **性能討論**: 比其對應的浮點 LUT 版本 (0.1271 秒) 更快，這是因為整數運算通常比浮點運算快。但仍然慢於原始的純 NumPy 浮點實現 (`traditional_bicubic.py`，0.0488 秒)，原因同上。
    *   **定點化意義**: 成功地將演算法轉換為定點運算流程，驗證了在保持高圖像品質的同時，可以避免複雜浮點運算。

## 6. 開發過程中的挑戰與解決方案

*   **Shell 環境問題 (`getcwd: cannot access parent directories`)**:
    *   在 `run_in_bash_session` 工具中執行 `cd` 命令後，有時會遇到此錯誤，導致後續命令失敗。
    *   解決方案：在執行 Python 腳本前，顯式地 `cd /app` (假定 `/app` 為專案根目錄)，確保 Python 解釋器在正確的工作目錄下啟動。
*   **Python 腳本中的相對路徑**:
    *   最初腳本內的路徑是相對於腳本所在位置 (`Bicubic-algorithm/`)。當從根目錄調用時，需要調整為相對於根目錄。
    *   解決方案：統一假設所有腳本從專案根目錄 (`/app`) 執行，並修改腳本內圖像文件的相對路徑（例如，從 `../lena_*.png` 改為 `lena_*.png`）。
*   **浮點 LUT 優化版本的初步性能不佳**:
    *   第一個 LUT 版本的 `optimized_bicubic_float.py` (使用 `np.vectorize` 包裝 LUT 查找函數) 執行非常慢 (約 0.22 秒)。
    *   原因：`np.vectorize` 主要是為了方便，其性能遠不如直接的 NumPy 數組操作。
    *   解決方案：修改 `contributions_lut` 函數，實現完全向量化的 LUT 索引計算和查表，避免使用 `np.vectorize`。這使得浮點 LUT 版本時間降至 0.1271 秒。
*   **定點數四捨五入**:
    *   將浮點數轉換為定點數，以及將累加的定點結果還原為像素值時，需要正確的四捨五入以減少誤差。
    *   解決方案：`np.round()` 用於浮點到定點的轉換。對於定點到像素值的還原，採用 `floor(value / scale_factor + 0.5)` 的方式進行四捨五入。

## 7. 總結與未來展望

本專案成功地完成了對 Bicubic 內插演算法的優化探索（通過 LUT 簡化核心運算）和定點化實現。

*   **成果**:
    *   所有開發的演算法版本（優化浮點、定點）在 PSNR 指標上均達到了與傳統實現相當的水平，圖像品質得到了保證。
    *   定點版本 (`optimized_bicubic_fixed_point.py` 使用 Qx.10 權重) 證明了在顯著降低運算複雜性的同時，仍能保持高圖像保真度。
    *   「運算簡化」的核心是將 Bicubic 核函數的直接計算替換為查表，這為後續移植到不具備高效浮點運算單元或對功耗有嚴格要求的平台（如FPGA、嵌入式MCU）提供了切實可行的思路。

*   **Python 環境下的性能觀察**:
    *   一個重要的觀察是，在 Python 環境中，利用 NumPy 進行直接的、向量化的浮點數學運算 (`traditional_bicubic.py`) 展現了極高的效率。基於 LUT 的方法（無論是浮點還是定點），雖然在理論上簡化了運算類型，但在 Python 中由於引入了額外的數組操作（索引計算、查表等），其執行時間未能超越原始的 NumPy 優化版本。這再次凸顯了針對特定環境選擇合適優化策略的重要性。

*   **未來工作**:
    *   **參數化定點精度**: 可以將 `LUT_FRAC_BITS` 以及可能的累加器位寬作為參數，進一步分析不同定點精度對 PSNR 和運算複雜度的影響。
    *   **硬體/低階語言實現考量**: 本專案的定點版本為在 C/C++ 或 HDL 中實現該演算法提供了清晰的邏輯。可以進一步考慮如何優化內存訪問、並行性等。
    *   **其他插值演算法**: 可以將類似的優化和定點化思路應用於其他圖像插值演算法，如 Lanczos 插值等。
    *   **彩色圖像處理**: 目前所有實現均針對灰度圖像。可以擴展到彩色圖像，通常是分別處理 R, G, B 三個通道。

本專案為理解 Bicubic 演算法的內部細節、探索其優化途徑以及如何將其有效地轉換為定點實現提供了有益的實踐。
