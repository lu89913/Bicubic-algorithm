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

### 3.2. `src/traditional_bicubic.py`
*   **用途**：實現了標準的浮點 Bicubic 插值演算法。
*   **核心**：`cubic_kernel()` 函數根據可配置的參數 `a`（如 -0.5, -0.75, -1.0）計算三次卷積核的權重。`bicubic_resize()` 函數使用這些權重來執行圖像的縮放。
*   **角色**：作為一個理想的、數學上精確的參考基準，用於評估硬體友好型定點實現所引入的精度損失。

### 3.3. `src/hardware_friendly_bicubic.py`
*   **用途**：實現了為硬體優化的定點 Bicubic 插值演算法。
*   **核心**：
    *   **定點化**：所有浮點計算都轉換為定點整數計算。小數精度由全局變量 `F_BITS` 控制（當前推薦值為 10）。
    *   `cubic_kernel_fixed_point()`: 計算定點形式的卷積核權重。特別地，對於選定的 `a` 值（-0.5, -0.75, -1.0），此函數將與 `a` 相關的係數乘法操作替換為硬體上更高效的位移 (shift) 和整數加/減 (add/sub) 操作。對於其他 `a` 值，則採用標準的定點乘法。
    *   `bicubic_resize_fixed_point()`: 執行完整的圖像縮放流程，包括獲取鄰近像素、計算插值權重、以及混合像素值，全部使用定點算術。
*   **角色**：項目的核心優化成果，展示了如何在保持高圖像品質的前提下，使 Bicubic 演算法更適合硬體實現。

### 3.4. `images/` 目錄
*   **`gradient.png` (64x64)**: 一個簡單的灰階漸變圖像，用於早期和快速的演算法驗證及初步的 PSNR 基準測試。由 `create_test_image.py` 生成。
*   **`complex_test_image_256.png` (256x256)**: 一個包含多種視覺元素的灰階圖像（如清晰邊緣、線條、圓形、不同灰階區域、漸變），用於更全面地評估插值演算法在不同特徵上的表現，特別是視覺效果。由 `create_complex_test_image.py` 生成。
*   **`output/`**: 此目錄存放由 `compare_algorithms.py` 在處理 `complex_test_image_256.png` 時，由不同 Bicubic 實現（Pillow, 浮點, 定點）和不同 `a` 值生成的放大圖像。這些圖像對於主觀視覺比較非常重要。

### 3.5. `tests/` 目錄
*   包含針對 `src/` 目錄中兩個核心演算法的單元測試。
*   `test_traditional_bicubic.py`: 驗證浮點實現的輸出尺寸，並將其與 Pillow 庫的 `BICUBIC` 輸出（當 `a=-0.5` 時）進行比較，以確保基準的合理性。
*   `test_hardware_friendly_bicubic.py`:
    *   驗證定點實現的輸出尺寸。
    *   關鍵測試是將定點實現的輸出與浮點實現的輸出進行比較，計算 PSNR 和 MSE，以量化定點化和乘法簡化所引入的誤差。
    *   還包含一個 `test_memory_access_simulation()` 測試，用於模擬和驗證行緩衝區 (line buffer) 策略在減少主記憶體訪問次數方面的有效性。

### 3.6. `compare_algorithms.py`
*   **用途**：一個多功能的腳本，是進行演算法評估和比較的中心。
*   **功能**：
    1.  加載指定的測試圖像（`gradient.png` 或 `complex_test_image_256.png`）。
    2.  設置縮放因子。
    3.  調用三種 Bicubic 插值實現：
        *   Pillow 庫的 `BICUBIC`（作為通用參考）。
        *   `src/traditional_bicubic.py` 中的浮點實現。
        *   `src/hardware_friendly_bicubic.py` 中的定點實現。
    4.  針對預設的 `a` 值列表 (-0.5, -0.75, -1.0) 遍歷測試。
    5.  從 `hardware_friendly_bicubic.py` 導入 `F_BITS` 值，確保測試時使用的定點精度與設計一致。
    6.  計算並輸出各種比較下的 PSNR (峰值信噪比) 和 MSE (均方誤差)。
    7.  如果處理的是 `complex_test_image_256.png`，則將不同演算法和 `a` 值生成的結果圖像保存到 `images/output/` 目錄。
    8.  輸出對卷積核和插值計算步驟的定性複雜度分析。
    9.  輸出對記憶體訪問優化（行緩衝區）效果的量化分析。
*   **角色**：自動化了實驗和數據收集過程，其輸出是 `README.md` 中許多分析和結論的基礎。

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

經過參數 `a` 和定點小數精度 `F_BITS` 的探索，項目推薦使用以下配置以獲得最佳的圖像品質和硬體友好性：
*   **卷積核參數 `a = -0.5`**
*   **定點小數精度 `F_BITS = 10`**

在此配置下，硬體友好的定點 Bicubic 實現在 `gradient.png` 測試圖像上與浮點實現的輸出完全相同 (PSNR 為無限大)，在更複雜的 `complex_test_image_256.png` 上也達到了極高的 PSNR (約 64 dB)，表明精度損失非常小。同時，通過針對 `a=-0.5` 的係數乘法簡化（使用移位和加法），以及核心插值步驟的定點化，顯著降低了對複雜硬體運算單元（如浮點 DSP）的依賴。記憶體行緩衝策略也大幅減少了對主記憶體的帶寬需求。

---

希望這份概覽能幫助您更好地理解項目！
