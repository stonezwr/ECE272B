import numpy as np
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import keras
import tensorflow as tf
import matplotlib.pyplot as plt

classes = ['pass', 'total_loss', 'edge', 'nodule', 'crack', 'deform']


def preprocessing(samples, labels=None):
    global classes
    if labels is not None:
        le = LabelEncoder()
        le.fit(classes)
        labels = le.transform(labels)
        samples = samples.astype(np.float)
        labels = labels.astype(np.int)
        # train_x, validate_x, train_y, validate_y = model_s.train_test_split(samples, labels, test_size=0.2,
        #                                                                     random_state=3)
        #                                                                     #random_state=datetime.now().second)
        # return train_x, validate_x, train_y, validate_y
        return samples, labels
    else:
        return samples


def large_cnn(x_train, num_classes):
    model = Sequential()
    # add model layers
    model.add(Conv2D(32, (3, 3), padding='same', data_format="channels_last", input_shape=x_train.shape[1:]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model


def small_cnn(x_train, num_classes):
    model = Sequential()
    # add model layers
    model.add(Conv2D(32, (3, 3), padding='same', data_format="channels_last", input_shape=x_train.shape[1:]))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model


def mlp(x_train, num_classes):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(512))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(0.25))
    model.add(Dense(512))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(0.25))
    model.add(Dense(512))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(0.25))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model


# load data
train_x = np.expand_dims(np.load("train_data.npy"), axis=3)
train_y = np.load("train_label.npy")
test_x = np.expand_dims(np.load("test_data.npy"), axis=3)

# preprocessing label
train_x, train_y = preprocessing(train_x, train_y)
test_x = preprocessing(test_x)
train_y = keras.utils.to_categorical(train_y)

# init network
net = large_cnn(train_x, len(classes))
# init optimizer
opt = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
# compile network
net.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# init callbacks
earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
checkpoint = keras.callbacks.ModelCheckpoint(monitor='val_loss', filepath='weights.hdf5', verbose=1,
                                             save_best_only=True)

# init data generator for data augmentation
datagen = ImageDataGenerator(data_format="channels_last", validation_split=0.2, horizontal_flip=True,
                             vertical_flip=True, width_shift_range=0.2, height_shift_range=0.2, rotation_range=180)
traingen = datagen.flow(train_x, train_y, batch_size=10, subset='training')
validategen = datagen.flow(train_x, train_y, batch_size=10, subset='validation')

# start training
hist = net.fit_generator(generator=traingen, steps_per_epoch=len(traingen), epochs=200, shuffle=True,
                         validation_data=validategen, validation_steps=len(validategen), verbose=2,
                         callbacks=[earlystopping, checkpoint], workers=4)

# show the best performance
accuracies = hist.history['val_acc']
best_accuracy = max(accuracies)

epochs = np.arange(len(accuracies))
# plot learning curve
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Large CNN accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("accuracy_epoch.png")

print("validation accuracy: %.2f%%" % (best_accuracy * 100))

# predict testing set
predict_y = net.predict(test_x).argmax(axis=-1)
le = LabelEncoder()
le.fit(classes)
predict_y = le.inverse_transform(predict_y)
np.save("prediction.npy", predict_y)
