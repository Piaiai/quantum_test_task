from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input
from keras.layers import Conv2D, Conv2DTranspose, Dropout
from keras.layers import MaxPooling2D, concatenate, UpSampling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K



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
    drop3 = Dropout(0.2)(conv3)
    pool3 = MaxPooling2D((2, 2))(drop3)

    conv4 = build_conv_block(256, pool3)
    drop4 = Dropout(0.4)(conv4)
    pool4 = MaxPooling2D((2, 2))(drop4)

    conv5 = build_conv_block(512, pool4)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = build_conv_block(256, merge6)

    up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([drop3, up7], axis=3)
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