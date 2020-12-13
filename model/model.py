import csv
import cv2
import numpy as np
import os.path

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Flatten, Dense, Dropout
except:
    from keras.models import Sequential
    from keras.models import load_model
    from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Flatten, Dense, Dropout


CENTER_IMG_PATH_CSV_COLUMN = 0
LEFT_IMG_PATH_CSV_COLUMN = 1
RIGHT_IMG_PATH_CSV_COLUMN = 2
ANGLE_CSV_COLUMN = 3
IMAGE_SIZE = (60, 320, 1)

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
    for line in lines[1::2]:
        #print("Line ", line)
        steering_correction = 0.1
        measurement = float(line[ANGLE_CSV_COLUMN])
        for img_side in [CENTER_IMG_PATH_CSV_COLUMN, LEFT_IMG_PATH_CSV_COLUMN, RIGHT_IMG_PATH_CSV_COLUMN]:
            source_path = line[img_side]
            current_path = os.path.join(img_path, source_path.split('/')[-1])
            image = cv2.cvtColor(cv2.imread(current_path)[80:140,:,:], cv2.COLOR_BGR2GRAY) / 255.
            images.append(image)
            if (img_side == LEFT_IMG_PATH_CSV_COLUMN):
                measurement = measurement + steering_correction
            if (img_side == RIGHT_IMG_PATH_CSV_COLUMN):
                measurement = measurement - steering_correction
            measurements.append(measurement + 0.5)
            # naiive augmentation via flipping
            aug_image = cv2.flip(image, 1)
            aug_measurement = -1.0 * measurement
            images.append(aug_image)
            measurements.append(aug_measurement + 0.5)
        
    assert(len(images) == len(measurements))

    X_train = np.expand_dims(np.array(images), 3)
    y_train = np.array(measurements)
    
    assert(X_train[0].shape == IMAGE_SIZE)
    
    print("X_train contains {} samples".format(len(X_train)))
    return X_train, y_train

def build_behavioral_model():
    model = Sequential()
    model.add(AveragePooling2D(input_shape=IMAGE_SIZE, pool_size=(3, 3), strides=2, padding="valid"))
    #model.add(Conv2D(8, 3, input_shape=IMAGE_SIZE, activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
    model.add(Conv2D(8, 3, input_shape=IMAGE_SIZE, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="same"))
    model.add(Dropout(0.15))
    model.add(Conv2D(16, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="same"))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="same"))
    model.add(Conv2D(64, 3, activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="same"))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam')
    
    return model


if __name__=="__main__":
    model = build_behavioral_model()
    model.summary()

    X_train, y_train = load_augmented_train_data()
    print("X_train contains {} samples".format(len(X_train)))
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=30, batch_size=8)

    model.save('saved_model.h5')
    model.save_weights('saved_weights.h5')

    loaded = load_model('saved_model.h5')
    lw = model
    lw.load_weights('saved_weights.h5')
