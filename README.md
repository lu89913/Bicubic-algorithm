# 硬體友善型 Bicubic 插值演算法比較

## 1. 專案概覽

本專案旨在開發並比較多種 Bicubic 插值演算法的 Python 實現，重點在於針對硬體實現進行優化，同時分析其對圖像品質 (PSNR) 的影響。專案包含以下幾個關鍵部分：
1.  **傳統 Bicubic 插值演算法 (`traditional_bicubic.py`)**：基於常見的 MATLAB `imresize` 行為的 Python 轉譯版本。
2.  **優化版 Bicubic 演算法（浮點數運算，`optimized_bicubic_float.py`）**：其核心卷積係數表示為精確分數，以便於硬體實現（例如，使用移位器和加法器代替通用乘法器），但所有計算仍使用浮點數。
3.  **優化版 Bicubic 演算法（模擬定點數運算，`optimized_bicubic_fixed_point.py`）**：進一步模擬在硬體實現中會遇到的量化和溢出效應。
4.  **版本比較腳本 (`compare_all_versions.py`)**：用於評估這三個版本的效能。

主要目標是展示如何實現硬體友善的優化，並模擬在轉向定點數硬體設計時預期的圖像品質權衡。主要測試案例是將 256x256 像素的圖像放大至 512x512 像素。

## 2. 核心演算法檔案

*   **`traditional_bicubic.py`**:
    *   包含基準的 Bicubic 插值演算法。
    *   旨在模擬標準 Bicubic 實現的行為。
    *   包含 `imresize()` 函數，可以運行於 `'org'` (基於迴圈) 或 `'vec'` (向量化，此原始版本中功能受限) 模式。

*   **`optimized_bicubic_float.py`**:
    *   實現了使用浮點數運算的硬體友善型 Bicubic 核心 (`hardware_friendly_cubic`)。
    *   其卷積核係數在數學上等同於傳統核心，但表示為簡單分數（例如，1.5 被處理為 3/2）。
    *   提供了 `imresize_optimized_float()` 函數，具有穩健的 `'org'` 和 `'vec'` 模式。

*   **`optimized_bicubic_fixed_point.py`**:
    *   模擬使用定點數運算的硬體友善型 Bicubic 演算法。
    *   定義了定點數參數 (總位寬 `W`，小數位寬 `F`)。
    *   包含了用於定點數操作 (加法、乘法、飽和處理) 的輔助函數。
    *   其中的 `hardware_friendly_cubic_fixed_point` 核心和 `imresize_fixed_point` 函數均使用這些模擬的定點數進行運算。
    *   此版本有助於評估在硬體中因精度有限而導致的潛在圖像品質下降。(目前僅實現了 `vec` 模式)。

## 3. 比較腳本

*   **`compare_all_versions.py`**:
    *   從三個演算法檔案中導入相應的 `imresize` 函數。
    *   生成測試圖像 (灰階和彩色)。
    *   使用三種實現方式運行圖像放大。
    *   計算並報告：
        *   “優化版浮點”與“傳統版”之間的 PSNR。
        *   “優化版定點”與“傳統版”(或“優化版浮點”)之間的 PSNR。
        *   各版本的執行時間。

## 4. 依賴套件

*   Python 3.x
*   NumPy
*   scikit-image (用於計算 PSNR)

使用 pip 安裝依賴套件：
```bash
pip install numpy scikit-image
```

## 5. 如何運行比較

1.  確保已安裝 Python 及所需的依賴套件。
2.  在終端機中導航到專案目錄。
3.  運行比較腳本：
    ```bash
    python compare_all_versions.py
    ```

## 6. 預期輸出與結果摘要

`compare_all_versions.py` 腳本將輸出：
*   各演算法版本在測試圖像上的執行狀態和計時。
*   PSNR 值和計時的摘要表格。

**預期的 PSNR 觀察結果：**
*   **優化版浮點 vs. 傳統版**: PSNR 應極高 (如果以 `traditional_bicubic.py` 的 `org` 模式作為穩定參考，理想情況下為 `inf dB`)，表明在浮點環境中，係數的重新表示不會導致品質損失。
*   **優化版定點 vs. 傳統版/優化版浮點**: PSNR 可能會低於純浮點比較，顯示定點量化的影響。下降幅度取決於所選的定點位寬 (`W`, `F`)。一個好的結果是獲得仍可接受的高 PSNR (例如 >30-35 dB，視應用而定)。

此設置有助於清晰理解從基準演算法，到硬體感知浮點優化，再到模擬定點實現過程中的演算法權衡。

## 7. 專案概覽文檔

關於專案背景、演算法設計選擇、定點模擬細節以及更深入的結果分析，請參閱 `project_overview.md` (稍後亦將更新為中文)。
