# 優化 Bicubic 內插演算法及其定點化實現

## 1. 專案目標

本專案旨在完成以下目標：

1.  開發一個優化的 Bicubic 內插演算法 (`optimized_bicubic_float.py`)。
    *   要求：其峰值信噪比 (PSNR) 不低於傳統的 Bicubic 實現。
    *   要求：運算過程相較於直接的數學公式計算（如冪運算）得到大幅簡化，為後續定點化設計鋪路。
2.  將優化後的浮點演算法轉換為定點版本 (`optimized_bicubic_fixed_point.py`)。
    *   要求：轉換後的定點版本應保持與浮點版本相近的 PSNR。
3.  探索傳統 Bicubic 演算法的變種，以期進一步提升 PSNR。

## 2. 檔案結構

```
.
├── Bicubic-algorithm/
│   ├── traditional_bicubic.py       # 傳統 Bicubic 實現 (源自 GitHub，包含向量化和循環模式)
│   ├── optimized_bicubic_float.py   # 優化的浮點 Bicubic 實現 (基於 LUT)
│   ├── optimized_bicubic_fixed_point.py # 優化的定點 Bicubic 實現 (基於 LUT)
│   ├── modified_bicubic_variants.py # 測試不同 Bicubic 核變種的腳本
│   └── preprocess_image.py        # 圖像預處理腳本 (下載並降採樣 Lena 圖像)
├── lena_golden_512.png              # 原始 512x512 Golden 圖像
├── lena_downscaled_256.png          # 降採樣後的 256x256 輸入圖像
├── lena_traditional_bicubic_vec_512.png # traditional_bicubic.py (vec) 輸出
├── lena_optimized_bicubic_float_512.png # optimized_bicubic_float.py 輸出
├── lena_optimized_bicubic_fixed_512.png # optimized_bicubic_fixed_point.py 輸出
├── lena_variant_keys_a-0.75_512.png # modified_bicubic_variants.py (a=-0.75) 輸出 (最佳 PSNR)
├── lena_pillow_bicubic_512.png      # Pillow 庫 Bicubic 輸出 (參考)
├── readme.md                        # 本檔案：專案大綱和模擬結果
└── project_overview.md              # 詳細的演算法開發過程和分析
```

## 3. 執行環境與依賴

*   Python 3.x
*   NumPy: `pip install numpy`
*   Pillow (PIL): `pip install Pillow`
*   Requests: `pip install requests` (用於 `preprocess_image.py` 下載圖像)

## 4. 執行說明

1.  **圖像預處理 (若需要)：**
    如果專案根目錄下沒有 `lena_golden_512.png` 和 `lena_downscaled_256.png`，請先執行預處理腳本：
    ```bash
    python Bicubic-algorithm/preprocess_image.py
    ```
    (請確保執行此命令時，當前目錄為專案根目錄 `/app`。)

2.  **執行各種 Bicubic 實現：**
    所有演算法腳本都應從專案根目錄 (`/app`) 執行，例如：
    ```bash
    python Bicubic-algorithm/traditional_bicubic.py
    python Bicubic-algorithm/optimized_bicubic_float.py
    python Bicubic-algorithm/optimized_bicubic_fixed_point.py
    python Bicubic-algorithm/modified_bicubic_variants.py
    ```
    腳本會將處理後的 512x512 圖像保存在專案根目錄，並在控制台輸出相應的 PSNR 和執行時間（部分腳本）。

## 5. 模擬結果摘要

*   **基準圖像：** Lena 圖像 (輸入為 256x256，放大至 512x512，與 512x512 Golden 圖像比較)
*   **測試環境：** Python 3.12 (或沙箱內置版本), NumPy

### 5.1 最終優化及定點化結果總結

下表匯總了本專案開發的幾個關鍵 Bicubic 演算法版本的性能。`optimized_bicubic_fixed_point.py` 的最終版本採用了 Keys (a=-0.75) 核以獲得更佳 PSNR。

| 演算法版本                                            | PSNR (dB) | 執行時間 (秒) | 備註                                                     |
| :---------------------------------------------------- | :-------- | :------------ | :------------------------------------------------------- |
| `traditional_bicubic.py` (Keys, a=-0.5, vec mode)   | 34.1076   | 0.0488        | NumPy 直接計算 (基準)                                  |
| `optimized_bicubic_float.py` (Keys, a=-0.5, LUT)    | 34.1076   | 0.1271        | LUT 浮點實現 (運算簡化)                                |
| `optimized_bicubic_fixed_point.py` (Keys, a=-0.75, Q10 LUT) | **34.4135** | 0.1247        | **最佳PSNR硬體友好型** (LUT 定點, Qx.10 權重)          |
| `Pillow` (Image.BICUBIC, 內部實現未知)              | 34.1102   | (未單獨計時)  | Pillow 庫參考                                            |

*註：執行時間在 Python 環境下測得，僅供參考。LUT 版本的主要優勢在於運算結構的簡化，利於硬體實現。*

### 5.2 Bicubic 核函數變種探索 (`modified_bicubic_variants.py`)

為了探索進一步提升 PSNR 的可能性，我們對傳統 Bicubic 演算法中的三次卷積核（Keys 核）進行了參數 `a`（控制銳度）的調整，並與 Mitchell-Netravali 核進行了比較。

| 變種 (`modified_bicubic_variants.py`) | PSNR (dB) | 執行時間 (秒) | 備註                                     |
| :-------------------------------------- | :-------- | :------------ | :--------------------------------------- |
| Keys (a=-0.5, Catmull-Rom)              | 34.1076   | 0.2350        | 與 `traditional_bicubic.py` PSNR 一致    |
| **Keys (a=-0.75)**                      | **34.4137** | 0.0828        | **PSNR 顯著提升**                        |
| Keys (a=-1.0)                           | 34.3662   | 0.0164        | PSNR 仍有提升，但略低於 a=-0.75          |
| Mitchell-Netravali (B=1/3, C=1/3)       | 33.2296   | 0.0175        | PSNR 較低，更側重視覺平滑性            |

**變種實驗結論：**
*   調整 Keys 核的參數 `a` 至 **-0.75** 時，PSNR 相比傳統的 `a=-0.5` 有顯著提升 (從 34.1076 dB 到 34.4137 dB)。這表明適度的銳化對提升 Lena 圖像在此測試下的客觀品質有益。

## 6. 整體結論

1.  **PSNR 提升與硬體友好設計的平衡：**
    *   通過對 Keys 三次卷積核參數的探索，發現 `a=-0.75` 時能在 Lena 圖像上獲得最佳的 PSNR (約 34.41 dB)，相較於傳統的 `a=-0.5` (PSNR 約 34.11 dB) 有顯著提升。
    *   最終的 `optimized_bicubic_fixed_point.py` (採用 Keys, a=-0.75 核，Q10 定點權重) 成功地將此高 PSNR 變種進行了定點化，PSNR 幾乎無損 (34.4135 dB)，同時保持了基於 LUT 的硬體友好設計。

2.  **運算過程簡化（LUT方法）：**
    *   基於 LUT 的方法 (`optimized_bicubic_float.py` 和 `optimized_bicubic_fixed_point.py`) 將核心的 Bicubic 核函數（涉及冪運算）替換為查表操作。這符合了「運算過程大幅簡化」的目標，特別是對於硬體實現，可以避免複雜的浮點運算單元，轉而使用 ROM/RAM 和簡單的整數運算。
    *   雖然在 Python/NumPy 環境下，由於 NumPy 的高度優化，LUT 方法的執行速度未超越直接的 NumPy 數學運算，但其運算結構的轉變是關鍵的「簡化」。

3.  **定點化的有效性：**
    *   定點化實現 (`optimized_bicubic_fixed_point.py`) 證明了使用 Qx.10 格式的權重足以在保持與浮點版本幾乎相同圖像品質的同時，完成 Bicubic 插值。這對於資源受限的硬體平台至關重要。

總體而言，本專案不僅實現了對 Bicubic 演算法的運算簡化探索和定點化，更進一步通過調整核函數參數找到了能夠顯著提升圖像品質的變種，並成功將該高品質變種轉化為硬體友好的定點實現。這為在實際應用中部署兼具高性能和高效率的 Bicubic 插值器提供了有價值的方案。詳細的開發過程和更深入的分析請參閱 `project_overview.md`。
