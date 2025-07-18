import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np

# === CONFIG ===
COVER_DIR = Path(r"C:\Users\preet\Downloads\LSBMR\boss_256_0.4\cover")  # Folder with original cover images
OUTPUT_DIR = Path(r"C:\Users\preet\Downloads\LSBMR\boss_256_0.4\LSBMR_Embedded")
RESIZED_COVER_DIR = OUTPUT_DIR / "cover_128x128"
PAYLOADS = [0.1, 0.2, 0.4]
MESSAGE = "This is a secret message for steganography research."
RESIZE_SHAPE = (128, 128)

# === LSBMR Helper Functions ===
def text_to_bits(text):
    bits = ''.join(f'{ord(c):08b}' for c in text)
    return np.array([int(b) for b in bits], dtype=np.uint8)

def prepare_image(image_path, size=(128, 128)):
    img = Image.open(image_path).convert("L").resize(size)
    return img

def embed_lsbmr(cover_img, message_bits, payload_ratio):
    pixels = np.array(cover_img).flatten()
    max_capacity_bits = (len(pixels) // 2) * 2
    embed_bits_len = int(max_capacity_bits * payload_ratio)
    message_bits = message_bits[:embed_bits_len]
    if len(message_bits) % 2 != 0:
        message_bits = np.append(message_bits, 0)

    new_pixels = pixels.copy()
    idx = 0
    for i in range(0, len(message_bits), 2):
        if idx + 1 >= len(pixels):
            break
        p1, p2 = int(pixels[idx]), int(pixels[idx + 1])
        m1, m2 = message_bits[i], message_bits[i + 1]

        if (p1 % 2 == m1) and (((p1 + p2) % 2) == m2):
            pass
        elif (p1 % 2 != m1) and (((p1 + p2 + 1) % 2) == m2):
            p1 = p1 + 1 if p1 < 255 else p1 - 1
        elif (p1 % 2 != m1) and (((p1 + p2 - 1) % 2) == m2):
            p1 = p1 - 1 if p1 > 0 else p1 + 1
        else:
            if ((p1 + p2) % 2) != m2:
                p2 = p2 + 1 if p2 < 255 else p2 - 1

        new_pixels[idx], new_pixels[idx + 1] = p1, p2
        idx += 2

    stego_img = Image.fromarray(new_pixels.reshape(cover_img.size[1], cover_img.size[0]))
    return stego_img

def embed_and_save(image_path, message_bits, payload, resize_shape, out_dir, cover_out_dir):
    img = prepare_image(image_path, resize_shape)
    cover_save_path = cover_out_dir / f"cover_{image_path.stem}.png"
    img.save(cover_save_path)
    stego_img = embed_lsbmr(img, message_bits, payload)
    stego_save_path = out_dir / f"stego_{image_path.stem}.png"
    stego_img.save(stego_save_path)

# === Run Embedding ===
def parallel_embedding():
    message_bits = text_to_bits(MESSAGE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESIZED_COVER_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(COVER_DIR.glob("*"))[:MAX_IMAGES]

    for payload in PAYLOADS:
        payload_dir = OUTPUT_DIR / f"payload_{int(payload * 100)}"
        payload_dir.mkdir(parents=True, exist_ok=True)
        tasks = [
            (img_path, message_bits, payload, RESIZE_SHAPE, payload_dir, RESIZED_COVER_DIR)
            for img_path in image_paths
        ]
        with ThreadPoolExecutor(max_workers=8) as executor:
            executor.map(lambda args: embed_and_save(*args), tasks)

if __name__ == "__main__":
    parallel_embedding()
