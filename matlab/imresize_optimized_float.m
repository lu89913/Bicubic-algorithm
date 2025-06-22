function B = imresize_optimized_float(I, varargin)
% -------------------------------------------------------------------------
% imresize_optimized_float: 硬體友善型 Bicubic 插值演算法 (浮點數) - 鄰域修正
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
    [~, order] = sort(scale);
    I_double = im2double(I);
    B = I_double;
    for pass = 1:2
        dim = order(pass);
        in_len = size(B, dim);
        out_len = output_shape(dim);
        kernel_width = 4.0;
        [weights, indices] = contributions(in_len, out_len, scale(dim), @hardware_friendly_cubic, kernel_width);
        B = resize_along_dim(B, dim, weights, indices);
    end
    if isa(I, 'uint8')
        B = im2uint8(B);
    end
end

% --- 輔助函數 (Local Functions) ---

function f = hardware_friendly_cubic(x)
    % (此函數不變)
    absx = abs(x);
    absx2 = absx .* absx;
    absx3 = absx2 .* absx;
    f = zeros(size(x), 'like', x);
    cond1 = absx <= 1;
    num1 = (3 * absx3(cond1)) - (5 * absx2(cond1)) + 2;
    f(cond1) = num1 / 2.0;
    cond2 = (1 < absx) & (absx <= 2);
    num2 = (-absx3(cond2)) + (5 * absx2(cond2)) - (8 * absx(cond2)) + 4;
    f(cond2) = num2 / 2.0;
end

function [weights, indices] = contributions(in_length, out_length, scale, kernel, k_width)
    % **【此函數已修正】**
    if scale < 1
        h = @(x) scale * kernel(scale * x);
        kernel_width_eff = k_width / scale;
    else
        h = kernel;
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
    weights = h(dist);
    sum_weights = sum(weights, 2);
    weights = bsxfun(@rdivide, weights, sum_weights);
    weights(sum_weights == 0, :) = 0;
    aux = [1:in_length, in_length-1:-1:1];
    indices = aux(mod(indices-1, numel(aux)) + 1);
end

function out_img = resize_along_dim(img, dim, weights, indices)
    % (此函數不變)
    if dim == 2
        img = permute(img, [2 1 3]); 
    end
    output_dim1 = size(weights, 1);
    output_dim2 = size(img, 2);
    num_channels = size(img, 3);
    out_img_permuted = zeros(output_dim1, output_dim2, num_channels, 'like', img);
    for k = 1:num_channels
        for j = 1:output_dim2
            col_in = img(:, j, k);
            pixels_to_process = col_in(indices);
            products = weights .* pixels_to_process;
            interpolated_col = sum(products, 2);
            out_img_permuted(:, j, k) = interpolated_col;
        end
    end
    if dim == 2
        out_img = ipermute(out_img_permuted, [2 1 3]);
    else
        out_img = out_img_permuted;
    end
end
