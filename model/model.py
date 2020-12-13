import csv
import cv2
import numpy as np
import os.path

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Flatten, Dense, Input, Dropout

PATH_CSV_COLUMN = 0
ANGLE_CSV_COLUMN = 3
IMAGE_SIZE = (80, 320, 3)

def load_augmented_train_data(data_folder='../data', log_name='driving_log.csv', img_folder='IMG'):
    print("Loading data from {}".format(data_folder))
    lines = []
    log_filename = os.path.join(data_folder, log_name)
    with open(log_filename) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    print("Data log at {} contains {} records".format(log_filename, len(lines)))
    img_path = os.path.join(data_folder, img_folder)
    images = []
    measurements = []
    for line in lines:
        source_path = line[PATH_CSV_COLUMN]
        filename = source_path.split('\\')[-1]
        current_path = os.path.join(img_path, filename)
        image = cv2.imread(current_path)[80:,:,:]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.
        images.append(image)
        measurement = float(line[ANGLE_CSV_COLUMN])
        measurements.append(measurement + 0.5)
        # naiive augmentation via flipping
        aug_image = cv2.flip(image, 1)
        aug_measurement = -1.0 * measurement
        images.append(aug_image)
        measurements.append(aug_measurement + 0.5)
        
    assert(len(images) == len(measurements))
    print("X_train contains {} samples".format(len(images)))

    X_train = np.array(images)
    y_train = np.array(measurements)
    
    return X_train, y_train

def build_behavioral_model():
    model = Sequential()
    model.add(Input(shape=IMAGE_SIZE))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=2, padding="valid"))
    model.add(Conv2D(8, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
    model.add(Dropout(0.2))
    model.add(Conv2D(16, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="same"))
    model.add(Flatten())
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam')
    
    return model


if __name__=="__main__":
    model = build_behavioral_model()
    model.summary()

    X_train, y_train = load_augmented_train_data()

    model.fit(X_train, y_train, validation_split=0.15, shuffle=True, epochs=10)

    model.save('saved_model.h5')
    model.save_weights('saved_weights.h5')

    loaded = load_model('saved_model.h5')
    lw = model
    lw.load_weights('saved_weights.h5')
