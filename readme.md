# 優化 Bicubic 內插演算法及其定點化實現

## 1. 專案目標

本專案旨在完成以下目標：

1.  開發一個優化的 Bicubic 內插演算法 (`optimized_bicubic_float.py`)。
    *   要求：其峰值信噪比 (PSNR) 不低於傳統的 Bicubic 實現。
    *   要求：運算過程相較於直接的數學公式計算（如冪運算）得到大幅簡化，為後續定點化設計鋪路。
2.  將優化後的浮點演算法轉換為定點版本 (`optimized_bicubic_fixed_point.py`)。
    *   要求：轉換後的定點版本應保持與浮點版本相近的 PSNR。

## 2. 檔案結構

```
.
├── Bicubic-algorithm/
│   ├── traditional_bicubic.py       # 傳統 Bicubic 實現 (源自 GitHub，包含向量化和循環模式)
│   ├── optimized_bicubic_float.py   # 優化的浮點 Bicubic 實現 (基於 LUT)
│   ├── optimized_bicubic_fixed_point.py # 優化的定點 Bicubic 實現 (基於 LUT)
│   └── preprocess_image.py        # 圖像預處理腳本 (下載並降採樣 Lena 圖像)
├── lena_golden_512.png              # 原始 512x512 Golden 圖像
├── lena_downscaled_256.png          # 降採樣後的 256x256 輸入圖像
├── lena_traditional_bicubic_vec_512.png # traditional_bicubic.py (vec) 輸出
├── lena_optimized_bicubic_float_512.png # optimized_bicubic_float.py 輸出
├── lena_optimized_bicubic_fixed_512.png # optimized_bicubic_fixed_point.py 輸出
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
    ```
    腳本會將處理後的 512x512 圖像保存在專案根目錄，並在控制台輸出相應的 PSNR 和執行時間（部分腳本）。

## 5. 模擬結果摘要

*   **基準圖像：** Lena 圖像 (輸入為 256x256，放大至 512x512，與 512x512 Golden 圖像比較)
*   **測試環境：** Python 3.12, NumPy (版本視沙箱環境而定)

| 演算法版本                                   | PSNR (dB) | 執行時間 (秒) | 備註                                   |
| :------------------------------------------- | :-------- | :------------ | :------------------------------------- |
| `traditional_bicubic.py` (vec mode)        | 34.1076   | 0.0488        | 基於 NumPy 的直接數學運算              |
| `optimized_bicubic_float.py` (LUT-based)   | 34.1076   | 0.1271        | 基於 LUT 的浮點實現                    |
| `optimized_bicubic_fixed_point.py` (Q10) | 34.1079   | 0.1000        | 基於 LUT 的定點實現 (權重Qx.10)        |
| `Pillow` (Image.BICUBIC)                   | 34.1102   | (未單獨計時)  | Pillow 庫參考                          |

*註：執行時間可能因運行環境的細微差異而略有波動。*

## 6. 結論

1.  **PSNR 達成：**
    *   優化的浮點版本 (`optimized_bicubic_float.py`) 和定點版本 (`optimized_bicubic_fixed_point.py`) 均成功實現了與傳統 Bicubic 演算法及 Pillow 庫幾乎相同的 PSNR，滿足了圖像品質不降低的要求。

2.  **運算過程簡化：**
    *   通過引入預計算的查找表 (LUT)，`optimized_bicubic_float.py` 和 `optimized_bicubic_fixed_point.py` 將核心的 Bicubic 核函數（涉及冪運算）替換為查表操作。這符合了「運算過程大幅簡化」的目標，特別是考慮到後續的定點化以及潛在的硬體實現，其中複雜的浮點運算成本較高。
    *   在 Python/NumPy 環境下，由於 NumPy 自身的 C 語言級別優化，直接的向量化數學運算 (`traditional_bicubic.py`) 展現出更快的執行速度。然而，LUT 方法的「簡化」更多體現在運算類型的轉變上，使其更適合資源受限的環境。

3.  **定點化實現：**
    *   `optimized_bicubic_fixed_point.py` 成功將浮點 LUT 演算法轉換為定點運算（權重使用 Qx.10 格式）。PSNR 結果表明，在所選的定點精度下，圖像品質損失極小。
    *   定點版本的執行速度比其對應的浮點 LUT 版本略快，符合預期。

總體而言，專案成功地探索了 Bicubic 演算法的優化路徑（通過 LUT 簡化核心計算）及其定點化實現，並驗證了其在圖像品質方面的可行性。詳細的開發過程和更深入的分析請參閱 `project_overview.md`。
