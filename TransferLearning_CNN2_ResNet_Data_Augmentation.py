import keras
from matplotlib import pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

import Utilities
from Preprocessing import *

keras.backend.set_image_data_format('channels_first')


def resnet_cnn(num_classes=5):
    model = Sequential()

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(3, 100, 100))
    base_model.trainable = False
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def run_resnet_cnn(X_train, t_train, X_val, t_val, epochs=20, batch_size=16):
    model = resnet_cnn(num_classes=5)
    optimizer = 'adam'
    save_model = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=2)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(
        X_train,
        t_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, t_val),
        shuffle=True,
        verbose=1,
        callbacks=[save_model]
    )

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_accuracy'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    model.load_weights('best_model.h5')
    return model


if __name__ == '__main__':
    grayscale = False
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(
        base_path='./dataset',
        grayscale=grayscale,
        data_augmentation=True
    )

    t_train = to_categorical(y_train, num_classes=5)
    t_val = to_categorical(y_val, num_classes=5)

    model = run_resnet_cnn(X_train, t_train, X_val, t_val)
    y_val_predicted = np.argmax(model.predict(X_val), axis=1)
    print('Validation Set metrics')
    Utilities.display_metrics(y_val, y_val_predicted)

    y_test_predicted = np.argmax(model.predict(X_test), axis=1)
    print('Test Set metrics')
    Utilities.display_metrics(y_test, y_test_predicted)
