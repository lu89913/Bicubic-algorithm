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
    *   包含基準的 Bicubic 插值演算法。旨在模擬標準 Bicubic 實現的行為。
    *   `mode='org'` (基於迴圈) 通常作為較穩定的參考。`mode='vec'` (向量化) 在此原始版本中功能受限。

*   **`optimized_bicubic_float.py`**:
    *   實現了使用浮點數運算的硬體友善型 Bicubic 核心 (`hardware_friendly_cubic`)。係數表示為簡單分數。
    *   提供了 `imresize_optimized_float()` 函數，具有穩健的 `'org'` 和 `'vec'` 模式。

*   **`optimized_bicubic_fixed_point.py`**:
    *   模擬使用定點數運算的硬體友善型 Bicubic 演算法 (例如，卷積核內部 Q7.8，像素/權重 Q15.8/UQ16.8，總位寬16/24)。
    *   有助於評估在硬體中因精度有限而導致的潛在圖像品質下降。(目前僅實現了 `vec` 模式)。

## 3. 比較腳本

*   **`compare_all_versions.py`**:
    *   導入三個演算法檔案的 `imresize` 函數。
    *   生成測試圖像，運行圖像放大，計算並報告 PSNR 及執行時間。

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

## 6. 模擬結果摘要 (基於最近測試)

`compare_all_versions.py` 腳本的典型輸出顯示：

*   **優化版浮點 (`org`) vs. 傳統版 (`org`)**:
    *   PSNR 約為 **48-49 dB**。這表明兩個浮點實現在視覺上非常接近，儘管由於細微的實現差異，未能達到理論上的無窮大 PSNR。
*   **優化版定點 (`vec`) vs. 優化版浮點 (`vec`)**:
    *   對於 Q_Kernel=Q7.8, Q_Pixel=UQ16.8/Q15.8 的配置，PSNR 同樣約為 **48-49 dB**。這是一個積極的信號，表明選定的定點參數能夠在很大程度上保持 `Optimized Float` 版本的圖像品質，沒有引入顯著的額外精度損失。
*   **關於 `traditional_bicubic.py` 的 `vec` 模式**:
    *   其輸出與 `optimized_fixed_point.py` (vec) 比較時得到 `inf dB` 的 PSNR，這通常指示 `traditional_vec` 模式的輸出本身可能存在問題或與定點版本的輸出意外地“錯誤到一致”，因此不宜作為衡量定點版本絕對精度的標準。
*   **執行時間**:
    *   定點版本的 Python 模擬由於其詳細的元素級運算，執行時間遠高於浮點版本。這符合預期，因為模擬的重點是精度而非 Python 性能。

詳細的結果表格、定點參數說明以及更深入的分析，請參閱 `project_overview.md`。

## 7. 專案概覽文檔

關於專案背景、演算法設計選擇、定點模擬細節以及更深入的結果分析，請參閱 `project_overview.md`。
