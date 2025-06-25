import imageio.v2 as imageio
from PIL import Image
import numpy as np
import requests
from io import BytesIO

def download_image(url):
    """Downloads an image from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        img = Image.open(BytesIO(response.content))
        return img
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return None
    except IOError as e:
        print(f"Error opening image: {e}")
        return None

def preprocess_lena():
    """Downloads the Lena image, saves it as golden, and creates a downscaled version."""
    lena_url = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"

    print("Downloading Lena image...")
    golden_img = download_image(lena_url)

    if golden_img is None:
        print("Failed to download Lena image. Exiting.")
        return

    if golden_img.mode != 'L': # Ensure it's grayscale for simplicity, or convert if RGB
        print("Converting image to grayscale...")
        golden_img = golden_img.convert('L')

    if golden_img.size != (512, 512):
        print(f"Warning: Downloaded image size is {golden_img.size}, not 512x512. Resizing...")
        golden_img = golden_img.resize((512, 512), Image.BICUBIC)

    golden_path = "lena_golden_512.png"
    downscaled_path = "lena_downscaled_256.png"

    try:
        print(f"Saving Golden image (512x512) to {golden_path}...")
        golden_img.save(golden_path)
        print(f"Golden image saved to {golden_path}")

        print("Downscaling image to 256x256...")
        downscaled_img = golden_img.resize((256, 256), Image.BICUBIC) # Using BICUBIC for downscaling as well

        print(f"Saving downscaled image (256x256) to {downscaled_path}...")
        downscaled_img.save(downscaled_path)
        print(f"Downscaled image saved to {downscaled_path}")

        print("Image preprocessing complete.")

    except IOError as e:
        print(f"Error saving image: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    preprocess_lena()
