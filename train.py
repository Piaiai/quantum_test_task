import zipfile
import numpy as np
import os
from os.path import dirname, realpath
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input 
from keras.layers import Conv2D, Conv2DTranspose, Dropout 
from keras.layers import MaxPooling2D, concatenate, UpSampling2D 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

os.makedirs('stage1_train', exist_ok=True)

with zipfile.ZipFile(dirname(realpath(__file__)) + '\\stage1_train.zip', 'r') as zip_ref:
    zip_ref.extractall('stage1_train')

TRAIN_PATH = 'stage1_train\\'

def extract_data(train_path, img_width=128, img_height=128, channels=3):
    train_ids = os.listdir(train_path)
    X_train = np.zeros((len(train_ids), img_height, img_width, channels), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), img_height, img_width, 1), dtype=np.bool)

    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = imread(path + '\\images\\' + id_ + '.png')[:,:,:channels]
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        X_train[n] = img
        mask = np.zeros((img_height, img_width, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '\\masks\\'))[2]:
            mask_ = imread(path + '\\masks\\' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (img_height, img_width), mode='constant', 
                                        preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = mask

    return X_train, Y_train

X_train, Y_train = extract_data(TRAIN_PATH)

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return (2. * intersection + smooth)/(union + smooth)

def build_conv_block(filters, prev_layer, kernel_size=3):
    conv = Conv2D(filters, kernel_size, activation='relu',
                  padding='same', kernel_initializer='he_normal')(prev_layer)
    return Conv2D(filters, kernel_size, activation='relu',
                  padding='same', kernel_initializer='he_normal')(conv)

def build_unet(input_shape):
    inputs = Input(input_shape)

    conv1 = build_conv_block(32, inputs)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = build_conv_block(64, pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = build_conv_block(128, pool2)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = build_conv_block(256, pool3)
    pool4 = MaxPooling2D((2, 2))(conv4)

    conv5 = build_conv_block(512, pool4)


    up6 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = build_conv_block(256, merge6)

    up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = build_conv_block(128, merge7)

    up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = build_conv_block(64, merge8)

    up9 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = build_conv_block(32, merge9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    return Model(inputs=inputs, outputs=conv10)

model = build_unet((128, 128, 3))
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=[dice_coef])

earlystopper = EarlyStopping(patience=5, verbose=1)
weight_saver = ModelCheckpoint('nucleus.h5', monitor='val_dice_coef', save_best_only=True, save_weights_only=True)

history = model.fit(X_train, Y_train, validation_split=0.2, batch_size=16, epochs=50,
                    callbacks=[earlystopper, weight_saver]) 