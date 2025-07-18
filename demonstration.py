import numpy as np
from PIL import Image
from joblib import load
from scipy.signal import convolve2d

# === Load your trained model and preprocessors ===
model = load('final_model.joblib')
scaler = load('srm_scaler_all.joblib')
pca = load('srm_pca_all.joblib')

# === SRM Feature Extractor ===
class SRMFeatureExtractor:
    def __init__(self):
        self.filters = self._init_filters()

    def _init_filters(self):
        F = []
        F.append(np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]]))
        F.append(np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]]))
        F.append(np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]]))
        F.append(np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]]))
        return F

    def extract(self, image):
        features = []
        image = image.astype(np.float32)
        for f in self.filters:
            residual = convolve2d(image, f, mode='valid')
            hist, _ = np.histogram(residual, bins=20, range=(-5, 5), density=True)
            features.extend(hist)
        return np.array(features)

# === 1. Load your test image ===
# Replace this path with your own image!
img_path = r'C:\Users\preet\OneDrive\Desktop\Image Steganalysis\test_img.jpg'

# === 2. Preprocess same as training ===
# Convert to grayscale & resize to 128×128
img = Image.open(img_path).convert('L').resize((128, 128))
img_np = np.array(img)

# === 3. Extract SRM features ===
extractor = SRMFeatureExtractor()
features = extractor.extract(img_np).reshape(1, -1)

# === 4. Standardize & reduce ===
features_scaled = scaler.transform(features)
features_pca = pca.transform(features_scaled)

# === 5. Predict ===
pred = model.predict(features_pca)
label = 'Stego' if pred[0] == 1 else 'Cover'

print(f"✅ The uploaded image is classified as: {label}")
