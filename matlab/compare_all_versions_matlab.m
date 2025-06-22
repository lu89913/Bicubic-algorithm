% -------------------------------------------------------------------------
% compare_all_versions_matlab.m
%
%   此腳本用於比較 MATLAB 內建 imresize, 優化浮點數版本,
%   以及優化定點數版本的圖像品質 (PSNR)。
%
%   請將此檔案與 imresize_optimized_float.m 和
%   imresize_optimized_fixed_point.m 放在同一目錄下。
% -------------------------------------------------------------------------
clear; clc; close all;

% --- 測試參數 ---
input_size = [256, 256];
output_size = [512, 512];

% --- 產生或讀取測試影像 ---
fprintf('正在準備測試影像...\n');
% 使用 MATLAB 內建測試影像
try
    test_img_orig = imread('C:\Users\xyz\Desktop\碩二\matlab\image2\origin_picture\nuk.png'); % 'cameraman.tif' 'peppers.png'
    if size(test_img_orig, 3) > 1
        test_img_orig = rgb2gray(test_img_orig);
    end
    test_img_orig = imresize(test_img_orig, input_size);
catch
    fprintf('找不到測試影像 "cameraman.tif"，將使用隨機影像。\n');
    test_img_orig = uint8(randi([0 255], input_size));
end

fprintf('輸入影像尺寸: %d x %d\n', size(test_img_orig,1), size(test_img_orig,2));
fprintf('目標輸出尺寸: %d x %d\n', output_size(1), output_size(2));

% --- 執行三種版本的 Bicubic 插值 ---

% 1. MATLAB 內建 imresize (基準)
fprintf('\n1. 執行 MATLAB 內建 imresize (bicubic)...\n');
tic;
img_matlab_bicubic = imresize(test_img_orig, output_size, 'bicubic');
time_matlab = toc;
fprintf('   完成, 耗時: %.4f 秒\n', time_matlab);

% 2. 優化版 Bicubic (浮點數)
fprintf('2. 執行優化版 Bicubic (浮點數)...\n');
tic;
img_opt_float = imresize_optimized_float(test_img_orig, 'OutputSize', output_size);
time_opt_float = toc;
fprintf('   完成, 耗時: %.4f 秒\n', time_opt_float);

% 3. 優化版 Bicubic (定點數模擬)
fprintf('3. 執行優化版 Bicubic (定點數模擬)...\n');
tic;
img_opt_fixed = imresize_optimized_fixed_point(test_img_orig, 'OutputSize', output_size);
time_opt_fixed = toc;
fprintf('   完成, 耗時: %.4f 秒\n', time_opt_fixed);

% --- 計算 PSNR ---
fprintf('\n--- 圖像品質比較 (PSNR) ---\n');

% 優化浮點 vs. MATLAB 內建
psnr_float_vs_matlab = psnr(img_opt_float, img_matlab_bicubic);
fprintf('優化浮點數版 vs. MATLAB 內建: %.2f dB\n', psnr_float_vs_matlab);

% 優化定點 vs. MATLAB 內建
psnr_fixed_vs_matlab = psnr(img_opt_fixed, img_matlab_bicubic);
fprintf('優化定點數版 vs. MATLAB 內建: %.2f dB\n', psnr_fixed_vs_matlab);

% 優化定點 vs. 優化浮點 (關鍵比較)
psnr_fixed_vs_float = psnr(img_opt_fixed, img_opt_float);
fprintf('優化定點數版 vs. 優化浮點數版: %.2f dB\n', psnr_fixed_vs_float);


% --- 顯示結果 ---
figure('Name', 'Bicubic 插值結果比較');
subplot(2, 2, 1);
imshow(test_img_orig);
title(sprintf('原始影像 (%dx%d)', input_size(1), input_size(2)));

subplot(2, 2, 2);
imshow(img_matlab_bicubic);
title(sprintf('MATLAB imresize (bicubic)\n耗時: %.2fs', time_matlab));

subplot(2, 2, 3);
imshow(img_opt_float);
title(sprintf('優化浮點數版\nPSNR: %.2f dB, 耗時: %.2fs', psnr_float_vs_matlab, time_opt_float));

subplot(2, 2, 4);
imshow(img_opt_fixed);
title(sprintf('優化定點數版\nPSNR: %.2f dB, 耗時: %.2fs', psnr_fixed_vs_matlab, time_opt_fixed));

fprintf('\n比較完成。\n');
