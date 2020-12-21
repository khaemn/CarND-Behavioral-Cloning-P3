import csv
import cv2
import numpy as np
import os.path
import sklearn
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# In the workshop environment it is OK to have old Keras as a separate package,
# but in any modern setup Keras is already a part of Tensorflow.
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.utils import plot_model
    from tensorflow.keras.callbacks import EarlyStopping
    PLOT_MODEL=True
except:
    from keras.models import Sequential
    from keras.models import load_model
    from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Flatten, Dense, Dropout
    from keras.callbacks import EarlyStopping


CENTER_IMG_PATH_CSV_COLUMN = 0
LEFT_IMG_PATH_CSV_COLUMN =   1
RIGHT_IMG_PATH_CSV_COLUMN =  2
ANGLE_CSV_COLUMN = 3

IMAGE_SIZE = (90, 160, 3)
# The input image 160*320 is being cropped to 80*320 starting from
# the 60th pixel (e.g. slightly higher than the vertical center),
# to 140th pixel. The most part of top of the image and also botom 20
# pixels do not bring in any useful visual information, thus cropped out.

def read_data_log(data_folder='data', log_name='driving_log.csv',):
    print("Loading data from {}".format(data_folder))
    records = []
    log_filename = os.path.join(data_folder, log_name)
    with open(log_filename) as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # Skip the first line, as it is a header
        for line in reader:
            records.append(line)
    print("Data log at {} contains {} records"
            .format(log_filename, len(records)))
    return records


def generator(input_samples, img_path='data/IMG', batch_size=32):
    num_samples = len(input_samples)
    all_possible_pics = [CENTER_IMG_PATH_CSV_COLUMN,
                         LEFT_IMG_PATH_CSV_COLUMN,
                         RIGHT_IMG_PATH_CSV_COLUMN]
    only_center = [CENTER_IMG_PATH_CSV_COLUMN]
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(input_samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                measurement = float(batch_sample[ANGLE_CSV_COLUMN])
                pictures_to_take = only_center
                if (abs(measurement) < 0.05):
                    pictures_to_take = all_possible_pics
                for img_side in pictures_to_take:
                    source_path = batch_sample[img_side]
                    current_path = os.path.join(img_path,
                                                source_path.split('/')[-1])
                    image = cv2.imread(current_path)            # load
                    image = image[60:150,::2,:]                 # crop to only see section with road
                    image = cv2.cvtColor(image, 
                                         cv2.COLOR_BGR2RGB)     # to rgb
                    image = (image / 255.5)                     # to range 0 .. 1
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
                    angles.append(measurement + 0.5)
                    # Naiive augmentation via flipping both the image and steering angle:
                    aug_image = cv2.flip(image, 1)
                    #cv2.imshow("image",cv2.cvtColor(((image+1.)*127.5).astype(np.uint8), cv2.COLOR_BGR2RGB))
                    #cv2.imshow("aug_image",cv2.cvtColor(((aug_image+1.)*127.5).astype(np.uint8), cv2.COLOR_BGR2RGB))
                    #cv2.waitKey(1000)
                    #quit()
                    aug_measurement = -1.0 * measurement
                    # Append this image and corresponding steering angle
                    images.append(aug_image)
                    angles.append(aug_measurement + 0.5)
            assert(len(images) == len(angles))
            X_train = np.array(images)
            y_train = np.array(angles)
            X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
            assert(X_train[0].shape == IMAGE_SIZE)
            assert(len(X_train) == len(y_train))
            assert(y_train[0].shape == ()) # must be a scalar with empty shape
            yield X_train, y_train


def build_behavioral_model():
    ''' Builds a Keras Sequential model with necessary architecture '''
    model = Sequential()
    # As the input image is highly pixelated and contains a lot of sharp
    # small details (which are not relevant to the steering), an average
    # pooling would blur and soften it, providing less noise signal to
    # the model. Also, not the whole image resolution is crucial, so this
    # layer also shrinks it a bit.
    #model.add(AveragePooling2D(input_shape=IMAGE_SIZE, pool_size=(3, 3),
    #          strides=1, padding="valid"))

    # 4 convolution-maxpooling segments; dropout after the very first one
    # seems the most common solution among small conv nets. 
    model.add(Conv2D(8, 3, input_shape=IMAGE_SIZE, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(16, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))

    model.add(Conv2D(24, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))

    model.add(Conv2D(48, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))

    model.add(Conv2D(96, 3, activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))

    model.add(Flatten())
    model.add(Dropout(0.25))
    
    # Dense layers to make decisions and the output neuron.
    model.add(Dense(96, activation='sigmoid'))
    model.add(Dense(64, activation='sigmoid'))
    # The output is sigmoid with range 0..1, but then the steering angle is
    # converted to a real one by subtracting 0.5.
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam')
    
    return model

def plot_history(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__=="__main__":
    model = build_behavioral_model()
    model.summary()
    
    if (PLOT_MODEL):
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        
    samples = read_data_log()[::1]
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    
    batch_size = 16
    
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)
    
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Batch size and number of epochs can be adjusted.
    # The parameters below are based on some local experiments.
    history = model.fit_generator(train_generator, \
            steps_per_epoch=math.ceil(len(train_samples)/batch_size), \
            validation_data=validation_generator, \
            validation_steps=math.ceil(len(validation_samples)/batch_size), \
            epochs=50, verbose=1, callbacks=[es])

    # Due to different Keras versions on a training machine and at the
    # Udacity workspace, it is impossible to save and load the whole
    # model as .h5 file -- it fails across different versions.
    # Luckily, the weight format remains the same, and so it is possible
    # to just re-create the model and then load pretrained weights.
    # See `restore_weights_to_model.py` for details.
    model.save_weights('saved_weights.h5')
    model.save('saved_model.h5')
    
    plot_history(history)
