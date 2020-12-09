import csv
import cv2
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Flatten, Dense, Input, Dropout

PATH_CSV_COLUMN = 0
ANGLE_CSV_COLUMN = 3
IMAGE_SIZE = (80, 320, 3)

lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
for line in lines[:100]:
    source_path = line[PATH_CSV_COLUMN]
    filename = source_path.split('\\')[-1]
    current_path = '../data/IMG/' + filename
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

X_train = np.array(images)
y_train = np.array(measurements)

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
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="same"))
model.add(Flatten())
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam')

model.summary()

model.fit(X_train, y_train, validation_split=0.15, shuffle=True, epochs=10)

model.save('saved_model.h5')

loaded = load_model('saved_model.h5')
