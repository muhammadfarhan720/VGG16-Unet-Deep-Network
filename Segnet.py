import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt


# Load the numpy files
x_train = np.load('/home/dprayati/Downloads/DataPreprocessing/x_train.npy')
x_test = np.load('/home/dprayati/Downloads/DataPreprocessing/x_test.npy')
x_val = np.load('/home/dprayati/Downloads/DataPreprocessing/x_validation.npy')
y_train = np.load('/home/dprayati/Downloads/DataPreprocessing/y_train.npy')
y_test = np.load('/home/dprayati/Downloads/DataPreprocessing/y_test.npy')
y_val = np.load('/home/dprayati/Downloads/DataPreprocessing/y_validation.npy')

y_train = to_categorical(y_train, num_classes=2)
y_val = to_categorical(y_val, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

def segnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    drop5 = Dropout(0.5)(conv5)
    # Decoder
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    up9 = UpSampling2D(size=(2, 2))(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv10 = Conv2D(2, 1, activation='softmax')(conv9)
    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

input_shape = x_train.shape[1:]
num_classes = 2  # For road and non-road classes
model = segnet(input_shape, num_classes)

# Train the model
epochs = 10
batch_size = 64
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('segnet_model.h5', verbose=1, save_best_only=True)
#history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                    #validation_data=(x_val, y_val),
                    #callbacks=[earlystopper, checkpointer])

model.load_weights('segnet_model.h5')



# Calculate the IoU (Intersection over Union)
y_pred = model.predict(x_test)
y_pred_argmax = np.argmax(y_pred, axis=3)
y_test_argmax = np.argmax(y_test, axis=3)
intersection = np.logical_and(y_test_argmax, y_pred_argmax)
union = np.logical_or(y_test_argmax, y_pred_argmax)


score = model.evaluate(x_test, y_test, batch_size=batch_size)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


# Calculate the IoU (Intersection over Union) for each class
iou_class1 = np.sum(intersection[:,:,0]) / np.sum(union[:,:,0])
iou_class2 = np.sum(intersection[:,:,1]) / np.sum(union[:,:,1])

# Compute the mean IOU across all classes
mean_iou = (iou_class1 + iou_class2) / 2.0

print('Mean IoU:', mean_iou)

n = 5
plt.figure(figsize=(20, 8))
for i in range(n):
    # Display the original image
    ax = plt.subplot(3, n, i+1)
    plt.imshow(x_test[i])
    plt.title('Original Image')
    plt.axis('off')
    
    # Display the ground truth mask
    ax = plt.subplot(3, n, i+n+1)
    plt.imshow(np.argmax(y_test[i], axis=-1))
    plt.title('Ground Truth Mask')
    plt.axis('off')
    
    # Display the segmented output image
    ax = plt.subplot(3, n, i+2*n+1)
    plt.imshow(np.argmax(y_pred[i], axis=-1))
    plt.title('Segmented Output')
    plt.axis('off')
plt.show()







