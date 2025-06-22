function B = imresize_optimized_fixed_point(I, varargin)
% -------------------------------------------------------------------------
% imresize_optimized_fixed_point: 硬體友善型 Bicubic 插值 (定點模擬) - 鄰域修正
% -------------------------------------------------------------------------
    % ... (主函數部分不變) ...
    p = inputParser;
    addRequired(p, 'I', @isnumeric);
    addParameter(p, 'Scale', 0, @isnumeric);
    addParameter(p, 'OutputSize', [0 0], @isnumeric);
    parse(p, I, varargin{:});
    in_shape = [size(I, 1), size(I, 2)];
    if p.Results.Scale > 0
        scale = [p.Results.Scale, p.Results.Scale];
        output_shape = ceil(scale .* in_shape);
    elseif all(p.Results.OutputSize > 0)
        output_shape = p.Results.OutputSize;
        scale = output_shape ./ in_shape;
    else
        error('請提供 "Scale" 或 "OutputSize" 參數。');
    end
    FP.W_Kernel = 16; FP.F_Kernel = 8;
    FP.MIN_Kernel = -(2^(FP.W_Kernel - 1));
    FP.MAX_Kernel = (2^(FP.W_Kernel - 1)) - 1;
    FP.W_Pixel = 24; FP.F_Pixel = 8;
    FP.MIN_Pixel = -(2^(FP.W_Pixel - 1));
    FP.MAX_Pixel = (2^(FP.W_Pixel - 1)) - 1;
    [~, order] = sort(scale);
    B = I;
    for pass = 1:2
        dim = order(pass);
        in_len = size(B, dim);
        out_len = output_shape(dim);
        kernel_width = 4.0;
        [weights_fixed, indices] = contributions_fixed(in_len, out_len, scale(dim), @hardware_friendly_cubic_fixed, kernel_width, FP);
        B = resize_along_dim_fixed(B, dim, weights_fixed, indices, FP);
    end
end

% --- 定點數輔助函數 (Local Functions) ---

function [weights, indices] = contributions_fixed(in_length, out_length, scale, kernel_fixed, k_width, FP)
    % **【此函數已修正】**
    if scale < 1
        h_float = @(x) scale * fixed_to_float(kernel_fixed(scale * x, FP), FP.F_Kernel);
        kernel_width_eff = k_width / scale;
    else
        h_float = @(x) fixed_to_float(kernel_fixed(x, FP), FP.F_Kernel);
        kernel_width_eff = k_width;
    end
    u = (1:out_length)' / scale + 0.5 * (1 - 1 / scale);
    
    % ---【修正點】---
    % 使用更穩健的方式計算左邊界，並固定 P=4
    left = floor(u) - 1;
    P = 4;
    % ---【修正結束】---

    indices_relative = (0:P-1);
    indices = bsxfun(@plus, left, indices_relative);
    dist = bsxfun(@minus, u, indices);
    float_weights_array = arrayfun(h_float, dist);
    sum_float_weights = sum(float_weights_array, 2);
    normalized_float_weights = bsxfun(@rdivide, float_weights_array, sum_float_weights);
    normalized_float_weights(sum_float_weights == 0, :) = 0;
    weights = arrayfun(@(x) float_to_fixed(x, FP.F_Pixel, FP.W_Pixel, true), normalized_float_weights);
    aux = [1:in_length, in_length-1:-1:1];
    indices = aux(mod(indices-1, numel(aux)) + 1);
end

% ... (其他所有輔助函數，如 resize_along_dim_fixed, fixed_multiply 等，維持不變) ...
% (請確保您檔案的其餘部分保持完整)
function val_sat = saturate(val, min_val, max_val)
    val_sat = max(min(val, max_val), min_val);
end
function fixed_val = float_to_fixed(val, F, W, is_signed)
    scaled_val = val * (2^F);
    fixed_val = int64(round(scaled_val));
    if is_signed
        min_v = -(2^(W - 1));
        max_v = (2^(W - 1)) - 1;
    else
        min_v = 0;
        max_v = (2^W) - 1;
    end
    fixed_val = saturate(fixed_val, min_v, max_v);
end
function float_val = fixed_to_float(fixed_val, F)
    float_val = double(fixed_val) / (2^F);
end
function res_fixed = fixed_multiply(a_fixed, b_fixed, W_out, F_a, F_b, F_out, is_signed)
    temp_product = int64(a_fixed) .* int64(b_fixed);
    shift_amount = (F_a + F_b) - F_out;
    if shift_amount > 0
        rounding_val = int64(bitshift(1, shift_amount - 1));
        temp_product = temp_product + rounding_val;
        res_fixed = bitshift(temp_product, -shift_amount);
    else
        res_fixed = bitshift(temp_product, -shift_amount);
    end
    if is_signed
        min_v = -(2^(W_out - 1));
        max_v = (2^(W_out - 1)) - 1;
    else
        min_v = 0;
        max_v = (2^W_out) - 1;
    end
    res_fixed = saturate(res_fixed, min_v, max_v);
end
function f_fixed = hardware_friendly_cubic_fixed(x_float, FP)
    x_fixed = float_to_fixed(abs(x_float), FP.F_Kernel, FP.W_Kernel, true);
    absx2_fixed = fixed_multiply(x_fixed, x_fixed, FP.W_Kernel, FP.F_Kernel, FP.F_Kernel, FP.F_Kernel, true);
    absx3_fixed = fixed_multiply(absx2_fixed, x_fixed, FP.W_Kernel, FP.F_Kernel, FP.F_Kernel, FP.F_Kernel, true);
    f_fixed = int64(0);
    if abs(x_float) <= 1
        term_x3 = fixed_multiply(3, absx3_fixed, FP.W_Kernel, 0, FP.F_Kernel, FP.F_Kernel, true);
        term_x2 = fixed_multiply(5, absx2_fixed, FP.W_Kernel, 0, FP.F_Kernel, FP.F_Kernel, true);
        const_2 = float_to_fixed(2.0, FP.F_Kernel, FP.W_Kernel, true);
        num1 = (term_x3 - term_x2) + const_2;
        f_fixed = round(double(num1) / 2.0);
    elseif abs(x_float) > 1 && abs(x_float) <= 2
        term_neg_x3 = -absx3_fixed;
        term_5x2 = fixed_multiply(5, absx2_fixed, FP.W_Kernel, 0, FP.F_Kernel, FP.F_Kernel, true);
        term_8x = fixed_multiply(8, x_fixed, FP.W_Kernel, 0, FP.F_Kernel, FP.F_Kernel, true);
        const_4 = float_to_fixed(4.0, FP.F_Kernel, FP.W_Kernel, true);
        num2 = (term_neg_x3 + term_5x2 - term_8x) + const_4;
        f_fixed = round(double(num2) / 2.0);
    end
    f_fixed = saturate(f_fixed, FP.MIN_Kernel, FP.MAX_Kernel);
end
function out_img_uint8 = resize_along_dim_fixed(img, dim, weights_fixed, indices, FP)
    if dim == 2
        img = permute(img, [2 1 3]);
    end
    img_fixed = arrayfun(@(x) float_to_fixed(double(x), FP.F_Pixel, FP.W_Pixel, false), img);
    output_dim1 = size(weights_fixed, 1);
    output_dim2 = size(img_fixed, 2);
    num_channels = size(img_fixed, 3);
    out_img_fixed_permuted = zeros(output_dim1, output_dim2, num_channels, 'int64');
    for k = 1:num_channels
        for j = 1:output_dim2
            col_in_fixed = img_fixed(:, j, k);
            for i = 1:output_dim1
                pixels_fixed = col_in_fixed(indices(i,:));
                w_fixed = weights_fixed(i,:);
                prod_fixed = fixed_multiply(pixels_fixed, w_fixed', FP.W_Pixel, FP.F_Pixel, FP.F_Pixel, FP.F_Pixel, true);
                sum_fixed = sum(prod_fixed);
                out_img_fixed_permuted(i, j, k) = saturate(sum_fixed, FP.MIN_Pixel, FP.MAX_Pixel);
            end
        end
    end
    if dim == 2
        out_img_fixed = ipermute(out_img_fixed_permuted, [2 1 3]);
    else
        out_img_fixed = out_img_fixed_permuted;
    end
    out_img_float = arrayfun(@(x) fixed_to_float(x, FP.F_Pixel), out_img_fixed);
    out_img_uint8 = uint8(round(saturate(out_img_float, 0, 255)));
end
