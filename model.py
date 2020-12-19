import csv
import cv2
import numpy as np
import os.path

# In the workshop environment it is OK to have old Keras as a separate package,
# but in any modern setup Keras is already a part of Tensorflow.
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.utils import plot_model
    PLOT_MODEL=True
except:
    from keras.models import Sequential
    from keras.models import load_model
    from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Flatten, Dense, Dropout


CENTER_IMG_PATH_CSV_COLUMN = 0
LEFT_IMG_PATH_CSV_COLUMN =   1
RIGHT_IMG_PATH_CSV_COLUMN =  2
ANGLE_CSV_COLUMN = 3

IMAGE_SIZE = (80, 320, 1)
# The input image 160*320 is being cropped to 80*320 starting from
# the 60th pixel (e.g. slightly higher than the vertical center),
# to 140th pixel. The most part of top of the image and also botom 20
# pixels do not bring in any useful visual information, thus cropped out.

def load_augmented_train_data(data_folder='../data',
                              log_name='driving_log.csv',
                              img_folder='IMG'):
    ''' Reads a driving log CSV with paths to images,
    then loads all the mentioned images, fipping each
    to achieve some data augmentation '''
    
    print("Loading data from {}".format(data_folder))
    lines = []
    log_filename = os.path.join(data_folder, log_name)
    with open(log_filename) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    print("Data log at {} contains {} records"
            .format(log_filename, len(lines)))
    img_path = os.path.join(data_folder, img_folder)
    images = []
    measurements = []
    for line in lines:
        measurement = float(line[ANGLE_CSV_COLUMN])
        # Note: as all the image are loaded into memory,
        # and also augmentation multiplies their volume twice,
        # I use datasets about 10k images. Larger datasets
        # could require refactring the data loader as Keras generator.
        for img_side in [CENTER_IMG_PATH_CSV_COLUMN,
                         LEFT_IMG_PATH_CSV_COLUMN,
                         RIGHT_IMG_PATH_CSV_COLUMN]:
            source_path = line[img_side]
            current_path = os.path.join(img_path,
                                        source_path.split('/')[-1])
            image = cv2.imread(current_path)            # load
            image = image[60:140,:,:]                   # crop
            image = cv2.cvtColor(image, 
                                 cv2.COLOR_BGR2GRAY)    # to grayscale
            image = image / 255.                        # to range 0 .. 1.0
            steering_correction = 0.15
            # For images from the left camera the car should drive more to the
            # right, and vice versa from images from right camera. Center
            # camera images do not require correction.
            if (img_side == LEFT_IMG_PATH_CSV_COLUMN):
                measurement = measurement + steering_correction
            if (img_side == RIGHT_IMG_PATH_CSV_COLUMN):
                measurement = measurement - steering_correction
            # Append this image and corresponding steering angle
            images.append(image)
            measurements.append(measurement + 0.5)
            # Naiive augmentation via flipping both the image and steering angle:
            aug_image = cv2.flip(image, 1)
            aug_measurement = -1.0 * measurement
            # Append this image and corresponding steering angle
            images.append(aug_image)
            measurements.append(aug_measurement + 0.5)
        
    assert(len(images) == len(measurements))

    X_train = np.expand_dims(np.array(images), 3)
    y_train = np.array(measurements)
    
    assert(X_train[0].shape == IMAGE_SIZE)
    assert(len(X_train) == len(y_train))
    assert(y_train[0].shape == [1])
    
    print("X_train contains {} samples".format(len(X_train)))
    return X_train, y_train

def build_behavioral_model():
    ''' Builds a Keras Sequential model with necessary architecture '''
    model = Sequential()
    # As the input image is highly pixelated and contains a lot of sharp
    # small details (which are not relevant to the steering), an average
    # pooling would blur and soften it, providing less noise signal to
    # the model. Also, not the whole image resolution is crucial, so this
    # layer also shrinks it a bit.
    model.add(AveragePooling2D(input_shape=IMAGE_SIZE, pool_size=(3, 3),
              strides=2, padding="valid"))

    # 4 convolution-maxpooling segments; dropout after the very first one
    # seems the most common solution among small conv nets. 
    model.add(Conv2D(8, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
    model.add(Dropout(0.2))

    model.add(Conv2D(16, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))

    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))

    model.add(Conv2D(32, 3, activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))

    model.add(Dropout(0.2))
    model.add(Flatten())
    
    # Two dense layer to make decisions and the output neuron.
    # One layer has driven worse than two.
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    # The output is sigmoid with range 0..1, but then the steering angle is
    # converted to a real one by subtracting 0.5.
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam')
    
    return model


if __name__=="__main__":
    model = build_behavioral_model()
    model.summary()
    
    if (PLOT_MODEL):
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    
    X_train, y_train = load_augmented_train_data()
    print("X_train contains {} samples".format(len(X_train)))
    
    # Batch size and number of epochs can be adjusted.
    # The parameters below are based on some local experiments.
    model.fit(X_train, y_train,
              validation_split=0.2,
              shuffle=True,
              epochs=12,
              batch_size=64)

    # Due to different Keras versions on a training machine and at the
    # Udacity workspace, it is impossible to save and load the whole
    # model as .h5 file -- it fails across different versions.
    # Luckily, the weight format remains the same, and so it is possible
    # to just re-create the model and then load pretrained weights.
    # See `restore_weights_to_model.py` for details.
    model.save_weights('saved_weights.h5')
