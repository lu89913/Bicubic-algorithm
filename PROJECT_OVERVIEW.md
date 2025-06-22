# 項目概覽：硬體友好型 Bicubic 插值演算法優化

## 1. 項目目標

本項目的核心目標是研究、設計並通過 Python 模擬驗證一種經過優化的 Bicubic 插值演算法。該演算法旨在保持高圖像品質的同時，實現硬體友好性，使其適合未來在 FPGA 等硬體平台上高效實現。主要關注點包括計算複雜度的降低（例如，用移位和加法替換乘法）和記憶體帶寬需求的減少。

## 2. 目錄結構和文件說明

```
.
├── images/                     # 存放測試圖像、生成腳本和輸出結果
│   ├── create_complex_test_image.py # 生成 complex_test_image_256.png 的腳本
│   ├── create_test_image.py    # 生成 gradient.png 的腳本
│   ├── complex_test_image_256.png # 256x256 複雜測試圖像
│   ├── gradient.png            # 64x64 簡單漸變測試圖像
│   └── output/                 # 存放 compare_algorithms.py 生成的插值結果圖像
├── src/                        # 核心演算法實現
│   ├── hardware_friendly_bicubic.py # 硬體友好型定點 Bicubic 實現
│   └── traditional_bicubic.py  # 傳統浮點 Bicubic 實現
├── tests/                      # 單元測試
│   ├── test_hardware_friendly_bicubic.py # 硬體友好型實現的測試
│   └── test_traditional_bicubic.py # 傳統浮點實現的測試
├── compare_algorithms.py       # 用於比較不同演算法性能和生成結果的腳本
├── README.md                   # 項目的主要文檔，包含成果總結和分析
└── PROJECT_OVERVIEW.md         # 本文件，提供項目結構和文件用途的快速指南
```

## 3. 主要文件詳解

### 3.1. `README.md`
*   **用途**：項目的入口文檔，詳細介紹了項目背景、目標、實施的優化策略（定點化、乘法簡化、記憶體訪問優化）、Python 模擬結果（包括 PSNR/MSE 表格）、以及對硬體實現潛在優勢的分析和最終結論。它更側重於項目的成果和發現。

### 3.2. `src/traditional_bicubic.py` (早期參考實現)
*   **用途**：實現了一個通用的浮點 Bicubic 插值演算法，允許配置卷積核參數 `a` 和邊界處理模式。
*   **角色**：在項目早期作為浮點參考。雖然通過將其邊界處理改為 `mode='edge'` 提高了與 MATLAB 的 PSNR（約 33.61 dB），但與 MATLAB `imresize` 的 `bicubic` 仍存在較大差異，表明 MATLAB 的實現有更特定的細節。目前主要被 `matlab_imresize_equivalent.py` 取代作為對 MATLAB 的主要比較基準。

### 3.3. `src/matlab_imresize_equivalent.py` (當前主要的 Python 浮點參考)
*   **用途**：一個從外部引入的 Python Bicubic 插值實現，其設計目標是**精確模擬 MATLAB `imresize(..., 'bicubic')` 函數（針對 `a=-0.5` 的情況）的行為**。它包含了特定的座標映射和邊界處理邏輯。
*   **核心**：其內部使用固定的 `a=-0.5` 三次卷積核。
*   **角色**：作為當前項目中與 MATLAB 進行比較的最準確的 Python 浮點參考。與 MATLAB `imresize` 的實際輸出達到了 **41.58 dB** 的高 PSNR。

### 3.4. `src/hardware_friendly_bicubic.py` (核心優化實現)
*   **用途**：實現了為硬體優化的定點 Bicubic 插值演算法。
*   **核心**：
    *   **定點化**：所有計算使用定點算術，小數精度 `F_BITS = 10`。
    *   `cubic_kernel_fixed_point()`: 計算定點卷積核權重。針對 `a=-0.5`（以及其他如 -0.75, -1.0 的值）進行了乘法簡化，使用移位和加/減法替代。
    *   `bicubic_resize_fixed_point()`: 執行定點圖像縮放。
*   **角色**：項目的核心成果。**其輸出（當 `a=-0.5, F_BITS=10`）與 `matlab_imresize_equivalent.py` 的輸出完全相同 (PSNR = Inf dB)，從而間接實現了與 MATLAB `imresize` 的高一致性 (PSNR ≈ 41.58 dB)。** 這證明了在保持與業界標準高度一致的圖像品質的同時，實現了硬體友好性。

### 3.5. `images/` 目錄
*   **`gradient.png` (64x64)**: 一個簡單的灰階漸變圖像，用於早期和快速的演算法驗證及初步的 PSNR 基準測試。由 `create_test_image.py` 生成。
*   **`complex_test_image_256.png` (256x256)**: 一個包含多種視覺元素的灰階圖像（如清晰邊緣、線條、圓形、不同灰階區域、漸變），用於更全面地評估插值演算法在不同特徵上的表現，特別是視覺效果。由 `create_complex_test_image.py` 生成。
*   **`output/`**: 此目錄存放由 `compare_algorithms.py` 在處理 `complex_test_image_256.png` 時，由不同 Bicubic 實現（Pillow, 浮點, 定點）和不同 `a` 值生成的放大圖像。這些圖像對於主觀視覺比較非常重要。

### 3.5. `tests/` 目錄
    *   包含針對 `src/` 目錄中主要演算法的單元測試。
    *   `test_traditional_bicubic.py`: 驗證 `traditional_bicubic.py`（早期參考）的輸出尺寸和與 Pillow 的比較。隨著 `traditional_bicubic.py` 角色的轉變，此測試的重要性相對降低，但仍可作為通用 Bicubic 核的行為參考。
*   `test_hardware_friendly_bicubic.py`:
    *   驗證定點實現的輸出尺寸。
        *   **關鍵測試**: 將定點實現的輸出與 `matlab_imresize_equivalent.py`（作為當前最準確的浮點參考）的輸出進行比較，以評估定點化的精度。
        *   包含記憶體訪問模擬測試。

### 3.7. `compare_algorithms.py`
*   **用途**：演算法評估和比較的中心腳本。
*   **功能**：
    1.  加載指定的測試圖像（默認為 `Cameraman.tif`，也可配置為 `complex_test_image_256.png` 等）。
    2.  設置縮放因子（默認為 2x）。
    3.  調用以下 Bicubic 插值實現：
        *   Pillow 庫的 `BICUBIC`（作為通用第三方庫參考）。
        *   `src/matlab_imresize_equivalent.py` (作為主要的 Python 浮點參考，模擬 MATLAB `imresize bicubic a=-0.5`)。
        *   `src/hardware_friendly_bicubic.py` (我們的定點實現，使用 `a=-0.5` 和 `F_BITS=10` 與上述參考進行比較)。
    4.  計算並輸出核心比較（`matlab_imresize_equivalent.py` vs `hardware_friendly_bicubic.py`）的 PSNR/MSE。
    5.  提示用戶提供 `matlab_imresize_equivalent.py` vs MATLAB 實際輸出的 PSNR/MSE，以完成對 MATLAB 的完整鏈路比較。
    6.  將所有生成的圖像保存到 `images/output/` 目錄。
    7.  輸出複雜度和記憶體訪問分析。
*   **角色**：自動化實驗、收集數據並促進與外部標準（如 MATLAB）的對比驗證。其輸出是 `README.md` 中分析和結論的直接來源。

## 4. 如何運行和測試

1.  **安裝依賴**:
    ```bash
    pip install numpy Pillow
    ```
2.  **生成測試圖像** (如果它們不存在):
    ```bash
    python images/create_test_image.py
    python images/create_complex_test_image.py
    ```
3.  **運行比較和評估**:
    ```bash
    python compare_algorithms.py
    ```
    這將使用 `complex_test_image_256.png` 進行2倍放大測試，並將輸出圖像保存到 `images/output/`。結果摘要將打印到控制台。要測試 `gradient.png`，可以修改 `compare_algorithms.py` 中的 `image_filename` 和 `scale_x/y` 變量。
4.  **運行單元測試**:
    ```bash
    python -m unittest discover tests
    ```
    或者單獨運行每個測試文件：
    ```bash
    python -m unittest tests.test_traditional_bicubic
    python -m unittest tests.test_hardware_friendly_bicubic
    ```

## 5. 主要發現和推薦配置

經過多次迭代和與 MATLAB `imresize` 行為的對比驗證，項目的主要發現和推薦配置如下：

*   **成功對齊 MATLAB**: 通過引入 `src/matlab_imresize_equivalent.py`（一個旨在精確模擬 MATLAB `imresize` 中 `bicubic` (`a=-0.5`) 選項的 Python 實現），我們成功地將 Python 浮點參考與 MATLAB 的實際輸出對齊到了 **41.58 dB** 的高 PSNR。
*   **定點實現的高保真度**: 我們的核心硬體友好型定點實現 `src/hardware_friendly_bicubic.py`（當配置為 `a=-0.5` 和 `F_BITS=10`）能夠**完美複製**上述 `matlab_imresize_equivalent.py` Python 浮點參考的輸出 (PSNR = Inf dB)。
*   **與 MATLAB 的高度一致性**: 綜合以上兩點，我們的硬體友好型定點 Bicubic 演算法的輸出結果，與 MATLAB `imresize` 這一廣泛使用的工業標準工具的輸出，達成了高度一致性 (PSNR ≈ 41.58 dB)。
*   **推薦配置**:
    *   **卷積核參數 `a = -0.5`**
    *   **定點小數精度 `F_BITS = 10`**
*   **核心優勢**: 此推薦配置不僅確保了與 MATLAB 標準的高圖像品質一致性，同時通過定點化和針對 `a=-0.5` 的乘法簡化（移位替換），顯著降低了硬體實現的複雜性。結合記憶體行緩衝策略，為開發高性能、低功耗的硬體 IP 核奠定了堅實基礎。

---

希望這份概覽能幫助您更好地理解項目！
