# 專案總覽：優化 Bicubic 內插演算法及其定點化

## 1. 專案背景與目標

本專案旨在對 Bicubic 圖像內插演算法進行優化和定點化轉換。主要目標包括：

1.  **開發優化浮點版本 (`optimized_bicubic_float.py`)**:
    *   在保持或提升圖像品質 (以 PSNR 衡量) 的前提下，大幅簡化 Bicubic 演算法核心的運算過程。簡化的目標是將複雜的數學運算（如浮點冪運算）替換為更適合硬體或低階語言實現的形式，例如查表。
    *   此版本不追求在 Python/NumPy 環境下超越高度優化的原生 NumPy 運算速度，而是側重於運算結構的改變。
2.  **開發定點版本 (`optimized_bicubic_fixed_point.py`)**:
    *   基於上述優化的浮點版本，將其轉換為定點數實現。
    *   確保定點版本能夠在可接受的精度損失範圍內（PSNR 與浮點版本相近）工作。
3.  **探索 Bicubic 變種以提升 PSNR (`modified_bicubic_variants.py`)**:
    *   研究調整傳統 Bicubic 核函數參數或使用不同三次核函數變體對 PSNR 的影響。
4.  **驗證與比較**:
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
    *   **Bicubic 核函數**: 採用標準的立方卷積核函數 (`cubic(x)`)，該函數通常隱含參數 `a = -0.5` (Catmull-Rom 樣條)。其數學表達式為：
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

*   **目標**: 結合第4章運算簡化的 LUT 思路和第7章變種探索中 Keys(a=-0.75) 核的 PSNR 優勢，開發一個最終的高品質、硬體友好的定點 Bicubic 實現。
*   **實現**:
    *   修改 `optimized_bicubic_fixed_point.py` 腳本。
    *   將其內部生成 `CUBIC_LUT_FLOAT` 時使用的 `CUBIC_PARAM_A` 全局參數從 `-0.5` 更改為 `-0.75`。
    *   保持定點策略不變：`LUT_FRAC_BITS = 10`，像素 `uint8` -> `int32` 計算，定點權重 `int32`，最終縮放、四捨五入、裁剪回 `uint8`。
    *   `contributions_fixed_point_weights(...)` 函數現在基於 `a=-0.75` 的核生成定點權重。
    *   `imresizevec_fixed_point(...)` 和 `imresize_optimized_fixed_point(...)` 的核心邏輯不變，僅是使用了新的權重。
*   **最終定點版本 (Keys, a=-0.75, Q10 LUT) 結果與分析**:
    *   **PSNR**: **34.4135 dB**。
    *   **與其浮點對應版本 (Keys, a=-0.75, float, from `modified_bicubic_variants.py`, PSNR 34.4137 dB) 比較**: 差異極小 (0.0002 dB)。這表明 `LUT_FRAC_BITS = 10` 的定點精度對於 `a=-0.75` 的核函數同樣能夠完美保持浮點版本的圖像品質。
    *   **與原 a=-0.5 定點版本 (PSNR 34.1079 dB) 比較**: PSNR 提升了約 0.3056 dB，與浮點版本中觀察到的提升一致。
    *   **執行時間**: 約 0.1247 秒。與之前的 a=-0.5 定點版本 (0.1000 秒) 和 a=-0.5 浮點LUT版本 (0.1271 秒) 在同一數量級。
    *   **定點化意義**: 成功將一個更高 PSNR 的 Bicubic 變種 (Keys, a=-0.75) 轉換為定點運算流程，驗證了其在保持高品質的同時，具備硬體實現的潛力。

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

## 7. Bicubic 核函數變種探索 (`modified_bicubic_variants.py`)

在完成了初步的 LUT 優化和定點化框架 (`optimized_bicubic_float.py` 和 `optimized_bicubic_fixed_point.py` 基於 `a=-0.5`) 之後，我們進行了此額外步驟，旨在探索是否可以通過調整 Bicubic 核函數的參數或類型來進一步提升 PSNR。

*   **動機**: 傳統 Bicubic 使用的 Keys 核通常設置參數 `a=-0.5` (Catmull-Rom)。不同的 `a` 值會影響圖像的銳利程度，而其他如 Mitchell-Netravali 的三次核則在不同的設計目標下（如視覺平滑性）被提出。我們希望找到一組參數或一個變種核，能夠在保持計算複雜度基本不變的情況下，提升 PSNR。
*   **實驗設計**:
    *   創建了 `modified_bicubic_variants.py`，該腳本允許靈活選擇和配置不同的三次卷積核。
    *   **測試的核函數及參數**:
        1.  **Keys 核 (Catmull-Rom)**: `a = -0.5` (作為與 `traditional_bicubic.py` 的校驗基準)。
        2.  **Keys 核 (銳化)**: `a = -0.75`。
        3.  **Keys 核 (更銳化)**: `a = -1.0`。
        4.  **Mitchell-Netravali 核**: `B = 1/3, C = 1/3`。
    *   測試方法與之前一致：使用 Lena 256x256 圖像放大至 512x512，並與 Golden 圖像比較 PSNR。所有測試均在 `vec` (向量化) 模式下進行。

*   **實驗結果**:

    | 變種 (`modified_bicubic_variants.py`) | PSNR (dB) | 執行時間 (秒) |
    | :-------------------------------------- | :-------- | :------------ |
    | Keys (a=-0.5, Catmull-Rom)              | 34.1076   | 0.2350        |
    | **Keys (a=-0.75)**                      | **34.4137** | 0.0828        |
    | Keys (a=-1.0)                           | 34.3662   | 0.0164        |
    | Mitchell-Netravali (B=1/3, C=1/3)       | 33.2296   | 0.0175        |

*   **結果分析**:
    *   **Keys 核 (a=-0.5)** 的 PSNR (34.1076 dB) 與 `traditional_bicubic.py` 的結果完全一致，驗證了測試代碼的正確性。其在此腳本中的執行時間 (0.2350 秒) 偏高，可能與 `lambda` 函數的輕微開銷或測試腳本的其他差異有關，但不影響相對 PSNR 的比較。
    *   **Keys 核 (a=-0.75)** 取得了最高的 PSNR，達到 **34.4137 dB**，相較於 `a=-0.5` 有約 0.3 dB 的顯著提升。這表明對於 Lena 圖像，適度的銳化（`a` 從 -0.5 調整到 -0.75）能夠改善圖像的客觀品質指標。
    *   **Keys 核 (a=-1.0)** 的 PSNR (34.3662 dB) 雖然也高於 `a=-0.5`，但略低於 `a=-0.75`。這暗示著過度銳化可能開始引入一些細微的振鈴或其他偽影，從而對 PSNR 產生輕微的負面影響，即使其執行速度非常快。
    *   **Mitchell-Netravali 核 (B=1/3, C=1/3)** 的 PSNR (33.2296 dB) 在所有測試中最低。這符合其設計初衷——該濾波器旨在實現模糊、振鈴和鋸齒偽影之間的良好視覺平衡，通常會產生比 Catmull-Rom 更平滑（視覺上可能更模糊）的結果，因此在 PSNR 這一客觀指標上可能不占優勢。

*   **變種實驗小結**:
    實驗表明，簡單地調整 Keys 三次卷積核的參數 `a`，特別是將其設置為 `-0.75`，可以在不顯著增加計算複雜性的前提下，有效地提升 Bicubic 插值在 Lena 圖像測試中的 PSNR。這為希望在傳統 Bicubic 基礎上獲得更高圖像品質的應用提供了一個簡單的改進方向。

## 7.5 額外實驗：Bicubic (a=-0.5) 加後處理銳化

為了探索另一種可能的優化路徑，我們嘗試了將一個較平滑的 Bicubic 插值結果 (使用 Keys 核, a=-0.5) 與一個簡單的後處理銳化濾波器結合起來。

*   **動機**: 使用較平滑的 `a=-0.5` 核可以減少插值本身引入的振鈴等偽影，然後通過一個可控的銳化步驟來提升圖像的清晰度，期望能找到一個比直接使用銳化型 Bicubic 核（如 a=-0.75）更好的平衡。
*   **方法**:
    1.  使用 `optimized_bicubic_fixed_point.py` 腳本（確保 `CUBIC_PARAM_A` 設置為 -0.5）生成 512x512 的插值圖像。此步驟的 PSNR 約為 34.1079 dB。
    2.  設計並實現一個簡單的 3x3 定點銳化濾波器 `apply_sharpen_filter_fixed_point`，選用核:
        ```
        K_sharpen = [[ 0, -1,  0],
                     [-1,  5, -1],
                     [ 0, -1,  0]]
        ```
        該核總和為1，係數均為小整數，易於定點實現。
    3.  將銳化濾波器應用於 Bicubic (a=-0.5) 的插值結果上。
    4.  在 `bicubic_plus_sharpen_fixed.py` 腳本中實現並測試此流程。

*   **實驗結果 (`bicubic_plus_sharpen_fixed.py`)**:
    *   Bicubic (a=-0.5, 定點) 插值時間: 約 0.15 - 0.18 秒。
    *   銳化濾波時間 (手動 NumPy 循環實現): 約 2.0 秒 (效率較低)。
    *   **最終 PSNR (Bicubic a=-0.5 定點 + Sharpen vs golden)**: **30.4884 dB**。

*   **結果分析**:
    *   **PSNR 大幅下降**: 應用所選的銳化核後，PSNR 從約 34.1 dB 急劇下降至約 30.5 dB。這表明該銳化核對於提升此場景下的 PSNR 是無效的，反而嚴重損害了圖像的客觀品質。銳化操作可能過度增強了噪點或產生了新的偽影。
    *   **手動卷積效率**: Python 循環實現的 3x3 卷積非常耗時，進一步說明了在 Python 中進行此類操作應優先使用 NumPy/SciPy 的內建優化函數（如果目標是執行速度）。但在本實驗中，由於 PSNR 結果不佳，未進一步優化其執行效率。

*   **此路徑小結**:
    「Bicubic (a=-0.5) + 特定後處理銳化」的組合方案在此次實驗中未能達到提升 PSNR 的目標。這強調了後處理步驟（尤其是銳化）的設計需要非常謹慎，並非所有增強視覺清晰度的操作都會帶來客觀品質指標的提升。相比之下，直接優化 Bicubic 核函數本身的參數（如採用 Keys, a=-0.75）被證明是更為直接和有效的提升 PSNR 的途徑。

## 8. 總結與未來展望

本專案圍繞 Bicubic 圖像內插演算法，成功地進行了多方面的探索與實現：

1.  **運算結構優化 (LUT 方法)**:
    *   通過 `optimized_bicubic_float.py`，驗證了使用預計算查找表 (LUT) 替代 Bicubic 核函數直接計算的可行性。此方法旨在簡化運算類型，將複雜的浮點冪運算等轉換為查表操作，這對於硬體實現或缺乏高效浮點支持的環境具有重要意義。
    *   雖然在 Python/NumPy 環境下，由於 NumPy 底層的高度優化，此 LUT 版本的執行速度並未超越直接使用 NumPy 數學函數的傳統 Bicubic 實現 (`traditional_bicubic.py`)，但它成功地改變了運算結構。

2.  **Bicubic 核函數變種探索以提升 PSNR**:
    *   通過 `modified_bicubic_variants.py`，對不同的三次卷積核（Keys 核的不同 `a` 參數、Mitchell-Netravali 核）進行了實驗評估。
    *   關鍵發現：將 Keys 核的參數 `a` 從傳統的 `-0.5` 調整為 `-0.75`，能夠在 Lena 圖像測試中將 PSNR 從約 34.11 dB 顯著提升至約 **34.41 dB**，而無需增加基礎 Bicubic 公式的計算複雜度。

3.  **高性能定點化實現**:
    *   結合上述兩點，最終的 `optimized_bicubic_fixed_point.py` 採用了 **Keys (a=-0.75) 核**，並基於 LUT 方法進行了定點化（權重使用 Qx.10 格式）。
    *   此最終定點版本實現了 **34.4135 dB** 的 PSNR，幾乎完美地保持了其浮點對應版本的圖像品質，同時顯著高於傳統 Bicubic (a=-0.5) 的 PSNR。
    *   這表明我們成功開發了一個兼具更高圖像品質和硬體實現友好性的 Bicubic 插值方案。

**核心成果**:
*   提出並驗證了一個改良的 Bicubic 變種 (Keys, a=-0.75)，其能夠提供更高的 PSNR。
*   成功將此改良變種通過 LUT 輔助的方式進行了定點化，並在定點精度 (Q10) 下保持了優異的圖像品質。
*   為需要在資源受限環境中實現高品質圖像縮放的應用，提供了一個具體的、經過驗證的演算法方案。

**未來工作建議**:
*   **定點精度細化分析**: 系統性地測試不同的 `LUT_FRAC_BITS` 和內部累加器位寬對 PSNR 和潛在硬體資源的影響，尋找最佳平衡點。
*   **直接定點計算 (針對 Keys, a=-0.75)**: 考慮到 Keys(a=-0.75) 核係數可以表示為 `N/4` 的形式，可以嘗試設計一個不依賴 LUT、而是利用移位操作優化乘法的直接定點計算版本，並與當前的 LUT 定點版本在複雜度和性能上進行比較。
*   **硬體實現原型**: 將 `optimized_bicubic_fixed_point.py` 中的邏輯（特別是針對 Keys, a=-0.75 的 LUT 定點版本）作為指導，在 Verilog/VHDL 或 HLS C++ 中進行原型設計，以評估實際的硬體資源消耗和運行速度。
*   **更廣泛的圖像集測試**: 在更多不同類型和內容的圖像上測試 Keys(a=-0.75) 核的普適性。
*   **彩色圖像支持**: 將目前的灰度圖像處理流程擴展至支持彩色圖像（例如，對 R, G, B 通道分別處理或轉換到 YCbCr 等色彩空間處理亮度通道）。

本專案為理解和改進經典的 Bicubic 插值演算法提供了有益的實踐經驗和具體的優化成果。
