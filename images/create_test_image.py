import numpy as np
from PIL import Image

def create_gradient_image(width, height, filename="images/gradient.png"):
    """Creates a simple grayscale gradient image and saves it."""
    array = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            array[y, x] = int(((x + y) / (width + height)) * 255)
    
    # Ensure the directory exists
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
            
    image = Image.fromarray(array, mode='L')
    image.save(filename)
    print(f"Saved test image to {filename}")

if __name__ == '__main__':
    create_gradient_image(64, 64)
