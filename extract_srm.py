import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.signal import convolve2d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import dump

# === SRM Extractor ===
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

def load_grayscale(path):
    return np.array(Image.open(path).convert('L'))

def extract_all():
    base_dir = r"C:\Users\preet\Downloads\LSBMR\boss_256_0.4\LSBMR_Embedded"
    cover_dir = os.path.join(base_dir, "cover_128x128")
    payload_dirs = {
        '10': os.path.join(base_dir, 'payload_10'),
        '20': os.path.join(base_dir, 'payload_20'),
        '40': os.path.join(base_dir, 'payload_40'),
    }
    extractor = SRMFeatureExtractor()

    X_all, y_all, meta_all = [], [], []

    cover_files = sorted(os.listdir(cover_dir))
    cover_feats = [extractor.extract(load_grayscale(os.path.join(cover_dir, f))) for f in tqdm(cover_files)]
    X_all.append(np.array(cover_feats))
    y_all.append(np.zeros(len(cover_feats)))
    meta_all += [(f, 0, 'cover') for f in cover_files]

    for payload, path in payload_dirs.items():
        files = sorted(os.listdir(path))
        feats = [extractor.extract(load_grayscale(os.path.join(path, f))) for f in tqdm(files)]
        X_all.append(np.array(feats))
        y_all.append(np.ones(len(feats)))
        meta_all += [(f, 1, f'payload_{payload}') for f in files]

    X = np.vstack(X_all)
    y = np.hstack(y_all)

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    pca = PCA(n_components=30).fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    np.save('X_features_pca_all.npy', X_pca)
    np.save('y_labels_all.npy', y)
    dump(scaler, 'srm_scaler_all.joblib')
    dump(pca, 'srm_pca_all.joblib')

if __name__ == "__main__":
    extract_all()
