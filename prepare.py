import os
import cv2
from tqdm import tqdm
import numpy as np
import shutil

path = 'images/'
original_path = os.path.join(path, "1024")
size = 256

originals = np.zeros((len(os.listdir(original_path)), 1024, 1024, 3), dtype=np.uint8)
resized = np.zeros((len(os.listdir(original_path)), size, size, 3), dtype=np.uint8)

for (i,name)  in enumerate(tqdm(os.listdir(original_path))):
    img_path = os.path.join(original_path, name)
    img = cv2.imread(img_path)
    originals[i] = img
    r = cv2.resize(img, (size, size))
    resized[i] = r


np.save(os.path.join(path, "originals.npy"), originals)
np.save(os.path.join(path, "resized.npy"), resized)