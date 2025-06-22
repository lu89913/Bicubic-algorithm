import numpy as np
from PIL import Image
import os
import math

def draw_line(arr, x0, y0, x1, y1, color):
    """Draws a simple line on a numpy array. Bresenham's line algorithm might be too complex for simple test."""
    # For simplicity, using a basic approach. Might not be perfect for all angles.
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    points = []
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    for p_x, p_y in points:
        if 0 <= p_y < arr.shape[0] and 0 <= p_x < arr.shape[1]:
            arr[p_y, p_x] = color

def create_complex_image(width, height, filename="images/complex_test_image_256.png"):
    """Creates a complex grayscale image with various features and saves it."""
    array = np.full((height, width), 128, dtype=np.uint8) # Mid-gray background

    # 1. Large black square (top-left)
    array[10:height//2 - 10, 10:width//2 - 10] = 0

    # 2. Large white square (top-right)
    array[10:height//2 - 10, width//2 + 10:width - 10] = 255

    # 3. Circle (bottom-left)
    center_x, center_y = width // 4, height * 3 // 4
    radius = min(width // 8, height // 8)
    for y in range(height):
        for x in range(width):
            if (x - center_x)**2 + (y - center_y)**2 < radius**2:
                array[y, x] = 64 # Dark gray
            elif (x - center_x)**2 + (y - center_y)**2 < (radius+1)**2 and \
                 (x - center_x)**2 + (y - center_y)**2 >= radius**2 :
                 array[y,x] = 0 # Black border for circle for sharpness

    # 4. Fine checkerboard/lines (bottom-right)
    patch_size = width // 8
    start_x, start_y = width * 3 // 4 - patch_size // 2, height * 3 // 4 - patch_size // 2
    for r in range(patch_size):
        for c in range(patch_size):
            abs_y, abs_x = start_y + r, start_x + c
            if 0 <= abs_y < height and 0 <= abs_x < width:
                if (r // 4 % 2 == 0 and c // 4 % 2 == 0) or \
                   (r // 4 % 2 != 0 and c // 4 % 2 != 0) : # coarse checker
                     if (r//2 % 2 == 0): # fine lines within checker
                        array[abs_y, abs_x] = 0 # Black lines
                     else:
                        array[abs_y, abs_x] = 255 # White lines
                else:
                    if (r//2 % 2 != 0):
                        array[abs_y, abs_x] = 30
                    else:
                        array[abs_y, abs_x] = 225


    # 5. Grayscale gradient strip (center)
    gradient_height = height // 8
    gradient_start_y = height // 2 - gradient_height // 2
    for y_offset in range(gradient_height):
        y = gradient_start_y + y_offset
        for x in range(10, width - 10): # Avoid edges of other squares
            array[y, x] = int((x - 10) / (width - 20) * 255)

    # 6. Diagonal lines
    draw_line(array, 5, 5, width - 5, height - 5, 255) # White diagonal
    draw_line(array, width - 5, 5, 5, height - 5, 0)   # Black diagonal

    # 7. Horizontal and Vertical lines
    mid_h_line_y = height // 2 + height // 4
    array[mid_h_line_y-1:mid_h_line_y+1, width//4 : width*3//4] = 20 # Dark gray horizontal

    mid_v_line_x = width // 2 + width // 4
    array[height//4 : height*3//4, mid_v_line_x-1:mid_v_line_x+1] = 230 # Light gray vertical

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    image = Image.fromarray(array, mode='L')
    image.save(filename)
    print(f"Saved complex test image to {filename}")

if __name__ == '__main__':
    create_complex_image(256, 256)
