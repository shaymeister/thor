"""
train_mask_detector.py

This script was built following this guide: COVID-19: Face Mask Detector with
OpenCV, Keras/Tensorflow, and Deep Learning on pyimagesearch.com
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from tqdm import tqdm

# hold all settings in dictionary
settings = {
    'AUGER_ROTATION_RANGE' : 20,
    'AUGER_ZOOM_RANGE' : 0.15,
    'AUGER_WIDTH_SHIFT_RANGE' : 0.2,
    'AUGER_HEIGHT_SHIFT_RANGE' : 0.2,
    'AUGER_SHEAR_RANGE' : 0.15,
    'AUGER_HORIZONTAL_FLIP' : True,
    'AUGER_FILL_MODE' : 'nearest',
    'TRAIN_INIT_LR' : 1e-4,
    'TRAIN_EPOCHS' : 20,
    'TRAIN_BATCH_SIZE' : 32,
    'TRAIN_TEST_SPLIT_TEST_SIZE' : 0.20,
    'TRAIN_TEST_SPLIT_RANDOM_STATE' : 42
}

def build_model():
    """load and compile the MobileNetV2 model"""

    # update user
    print('[INFO] Loading and Compiling Model')

    # loaded topless MobileNetV2 model
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(224, 224, 3))
    )

    # construct the head of the model
    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(7,7))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(128, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(2, activation="softmax")(head_model)

    # compile the base model and head models together
    model = Model(
        inputs=base_model.input,
        outputs=head_model
    )

    # freeze all layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # initialize Adam optimizer
    optimizer = Adam(
        lr=settings['TRAIN_INIT_LR'],
        decay=settings['TRAIN_INIT_LR'] / settings['TRAIN_EPOCHS']
    )

    # compile model
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    # update user
    print('[INFO] Loaded and Compiled Model')

    # return to runtime environment
    return model


def create_argument_parser():
    """created command-line argument parser"""

    # initialize argument parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument(
        '-d',
        '--dataset',
        required=True,
        type=str,
        action='store',
        help='path to mask dataset'
    )
    parser.add_argument(
        '-p',
        '--plot',
        required=False,
        type=str,
        action='store',
        default='plot.png',
        help='path to output poss/accuracy plot'
    )
    parser.add_argument(
        '-m',
        '--model',
        required=False,
        type=str,
        default='mask_detector.model',
        help='path to output face mask detector model'
    )

    # return parsed argument parser
    return parser.parse_args()


def load_dataset(path):
    """load all images and class from path"""

    # update user
    print('[INFO]: loading images from {}'.format(path))

    # find all images in path
    image_paths = list(paths.list_images(path))
    images = []
    labels = []

    # load all images in path
    for image_path in tqdm(image_paths):
        # extract the class label from filename
        label = image_path.split(os.path.sep)[-2]

        # load input image and preprocess
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        # update the data and labels lists
        images.append(image)
        labels.append(label)

    # convert images and labels to NumPy arrays
    images = np.array(images, dtype='float32')
    labels = np.array(labels)
    
    # update user
    print("[INFO]: loaded {} images from {}".format(len(images), path))

    # return images and labels
    return images, labels


def main(args):
    """initialize and control all script functionality"""
    
    # load all images and classes from the dataset
    images, labels = load_dataset(args.dataset)

    # perform one-hot encoding on the labels
    lb = LabelBinarizer()

    # fit labels
    labels = lb.fit_transform(labels)

    # convert labels to categorical
    labels = to_categorical(labels)

    # partition the dataset into training and testing splits
    train_x, test_x, train_y, test_y = train_test_split(
        images,
        labels,
        test_size=settings['TRAIN_TEST_SPLIT_TEST_SIZE'],
        stratify=labels,
        random_state=settings['TRAIN_TEST_SPLIT_RANDOM_STATE']
    )

    # construct the training image generator for data augmentation
    auger = ImageDataGenerator(
        rotation_range=settings['AUGER_ROTATION_RANGE'],
        zoom_range=settings['AUGER_ZOOM_RANGE'],
        width_shift_range=settings['AUGER_WIDTH_SHIFT_RANGE'],
        height_shift_range=settings['AUGER_HEIGHT_SHIFT_RANGE'],
        shear_range=settings['AUGER_SHEAR_RANGE'],
        horizontal_flip=settings['AUGER_HORIZONTAL_FLIP'],
        fill_mode=settings['AUGER_FILL_MODE']
    )

    # build model
    model = build_model()

    # train the head of the network
    new_model = model.fit(
        auger.flow(train_x, train_y, batch_size=settings['TRAIN_BATCH_SIZE']),
        steps_per_epoch=len(train_x)//settings['TRAIN_BATCH_SIZE'],
        validation_data=(test_x, test_y),
        validation_steps=len(test_x)//settings['TRAIN_BATCH_SIZE'],
        epochs=settings['TRAIN_EPOCHS']
    )

    # update user
    print("[INFO] Evaluating Network Performance...")

    # make predictions on testing set
    inferences = model.predict(test_x, batch_size=settings['TRAIN_BATCH_SIZE'])

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    inferences = np.argmax(inferences, axis=1)

    # show a nicely formatted classification report
    print(classification_report(test_y.argmax(axis=1), inferences,
	    target_names=lb.classes_))

    # save model
    print('[INFO] Saving Mask Detector Model')
    model.save(args.model)

    # plot the training loss and accuracy
    N = settings['TRAIN_EPOCHS']
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), new_model.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), new_model.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), new_model.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), new_model.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args.plot)

if __name__ == "__main__":
    # create command line argument parser
    args = create_argument_parser()

    # start the script with command line arguments
    main(args)