from PIL import Image
import numpy as np

# === CONFIG ===
IMG_PATH = r"C:\Users\preet\OneDrive\Desktop\Image Steganalysis\test_img_1(C).jpg"  # Your raw image
OUT_PATH =  r"C:\Users\preet\OneDrive\Desktop\Image Steganalysis\test_img_2(S).jpg"
MESSAGE = "This is a secret message for steganography research."
PAYLOAD = 0.2  # bits per pixel, e.g., 0.1, 0.2, 0.4

# === UTILS ===
def text_to_bits(text):
    bits = ''.join(f'{ord(c):08b}' for c in text)
    return np.array([int(b) for b in bits], dtype=np.uint8)

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

# === RUN ===
# 1. Load original image, grayscale & resize same as pipeline
cover_img = Image.open(IMG_PATH).convert("L").resize((128, 128))

# 2. Embed
message_bits = text_to_bits(MESSAGE)
stego_img = embed_lsbmr(cover_img, message_bits, PAYLOAD)

# 3. Save stego image
stego_img.save(OUT_PATH)

print(f"✅ New stego image saved at: {OUT_PATH}")
