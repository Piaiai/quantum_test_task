import os
import numpy as np 
from os.path import dirname, realpath
from skimage.io import imread
from skimage.transform import resize
import zipfile 
import cv2
from train import build_unet, build_conv_block

os.makedirs('stage1_test', exist_ok=True)

with zipfile.ZipFile('stage1_test.zip', 'r') as zip_ref:
    zip_ref.extractall('stage1_test')

TEST_PATH = 'stage1_test/'
model = build_unet((128, 128, 3))
model.load_weights('nucleus.h5')

test_ids = os.listdir(TEST_PATH)
X_test = np.zeros((len(test_ids), 128, 128, 3), dtype=np.uint8)

for n, id_ in enumerate(test_ids):
    path = TEST_PATH + id
    img = imread(path + '/images/' + id + '.png')[:,:,:3]
    img = resize(img, (128, 128), mode='constant', preserve_range=True)
    X_test[n] = img

preds_test = model.predict(X_test, verbose=1)

for prediction, id in zip(preds_test, test_ids):
    path = f'{TEST_PATH}{id}/images/{id}_segmented.png'
    cv2.imwrite(path, prediction) 

