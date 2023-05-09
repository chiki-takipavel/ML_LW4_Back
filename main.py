import scipy.io as sio
import numpy as np
import tensorflow as tf
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


TRAIN_FILE = "./dataset/train_32x32.mat"
TEST_FILE = "./dataset/test_32x32.mat"
MODEL_FILE = "model.tflite"
CLASSES_COUNT = 10
BALANCE_BORDER = 0.85
DATA_NAME = "X"
LABELS_NAME = "y"
HASHED_DATA_NAME = "X_hashed"
BATCH_SIZE = 64
INITIAL_LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 1e-6
DECAY_STEPS = 20000
DECAY_RATE = 0.9
EPOCHS = 50
EPOCHS_RANGE = range(EPOCHS)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(file_path):
    data = sio.loadmat(file_path)
    x = np.array(data[DATA_NAME])
    y = np.array(data[LABELS_NAME])
    x = np.moveaxis(x, -1, 0)
    y[y == 10] = 0

    return x, y


def remove_duplicates(data, labels):
    data_frame = pd.DataFrame({DATA_NAME: list(data), LABELS_NAME: list(labels)})
    data_bytes = [item.tobytes() for item in data_frame[DATA_NAME]]
    data_frame[HASHED_DATA_NAME] = data_bytes
    data_frame.sort_values(HASHED_DATA_NAME, inplace=True)
    data_frame.drop_duplicates(subset=HASHED_DATA_NAME, keep="first", inplace=True)
    data_frame.pop(HASHED_DATA_NAME)

    data_unique = np.array(list(data_frame[DATA_NAME].values))
    labels_unique = np.array(list(data_frame[LABELS_NAME].values))

    return data_unique, labels_unique


def show_classes_histogram(classes, counts):
    plt.figure()
    plt.bar(classes, counts)
    plt.show()
    logging.info("Histogram shown")


def check_classes_balance(labels):
    classes, counts = np.unique(labels, return_counts=True)

    max_images_count = max(counts)
    avg_images_count = sum(counts) / len(counts)
    balance_percent = avg_images_count / max_images_count

    show_classes_histogram(classes, counts)
    logging.info(f"Balance: {balance_percent:.3f}")
    if balance_percent > BALANCE_BORDER:
        logging.info("Classes are balanced")
    else:
        logging.info("Classes are not balanced")


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def prepare_data(data_train, data_test, to_grayscale=True):
    if to_grayscale:
        data_train = rgb2gray(data_train)
        data_test = rgb2gray(data_test)

        data_train = data_train.reshape((-1, 32, 32, 1))
        data_test = data_test.reshape((-1, 32, 32, 1))

    data_train = data_train / 255.0
    data_test = data_test / 255.0

    return data_train, data_test


def prepare_labels(labels_train, labels_test):
    labels_train = tf.keras.utils.to_categorical(labels_train, CLASSES_COUNT)
    labels_test = tf.keras.utils.to_categorical(labels_test, CLASSES_COUNT)

    return labels_train, labels_test


def split_test_val(data_test, labels_test):
    data_test, data_val, labels_test, labels_val = train_test_split(
        data_test,
        labels_test,
        test_size=0.5,
        random_state=42)

    return data_test, data_val, labels_test, labels_val


def create_model(image_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu',
            input_shape=(image_size, image_size, 1),
            kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(CLASSES_COUNT, activation='softmax')
    ])

    return model


def compile_model(model, data_train, labels_train, data_val, labels_val):
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=INITIAL_LEARNING_RATE,
        decay_steps=DECAY_STEPS,
        decay_rate=DECAY_RATE,
        staircase=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=6,
        verbose=1,
        min_lr=MIN_LEARNING_RATE
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )

    model_history = model.fit(
        x=data_train,
        y=labels_train,
        validation_data=(data_val, labels_val),
        epochs=EPOCHS,
        callbacks=[reduce_lr],
        verbose=1
    )

    return model_history


def evaluate_model(model, data_test, labels_test):
    test_loss, test_acc = model.evaluate(data_test, labels_test)
    logging.info(f"Test accuracy: {test_acc}; test loss: {test_loss}.")


def get_statistics(model_history):
    loss = model_history.history["loss"]
    accuracy = model_history.history["accuracy"]
    validation_loss = model_history.history["val_loss"]
    validation_accuracy = model_history.history["val_accuracy"]

    return loss, accuracy, validation_loss, validation_accuracy


def show_result_plot(loss, accuracy, validation_loss, validation_accuracy):
    plt.figure(figsize=(20, 14))

    plt.subplot(1, 2, 1)
    plt.title("Training and Validation Loss")
    plt.plot(EPOCHS_RANGE, loss, label="Train Loss")
    plt.plot(EPOCHS_RANGE, validation_loss, label="Validation Loss", linestyle="dashed")
    plt.legend(loc="upper right")

    plt.subplot(1, 2, 2)
    plt.title("Training and Validation Accuracy")
    plt.plot(EPOCHS_RANGE, accuracy, label="Train Accuracy")
    plt.plot(EPOCHS_RANGE, validation_accuracy, label="Validation Accuracy", linestyle="dashed")
    plt.legend(loc="lower right")

    plt.show()
    logging.info("Plot shown")


def save_model_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(MODEL_FILE, "wb") as f:
        f.write(tflite_model)

    logging.info("File saved")


def main(dataset_name="mnist"):
    if dataset_name == "mnist":
        (data_train, labels_train), (data_test, labels_test) = tf.keras.datasets.mnist.load_data()
        image_size = 28
        to_grayscale = False
    else:
        data_train, labels_train = load_data(TRAIN_FILE)
        data_test, labels_test = load_data(TEST_FILE)
        image_size = 32
        to_grayscale = True

    data_train, labels_train = remove_duplicates(data_train, labels_train)
    data_test, labels_test = remove_duplicates(data_test, labels_test)

    check_classes_balance(labels_train)
    check_classes_balance(labels_test)

    data_train, data_test = prepare_data(data_train, data_test, to_grayscale)
    labels_train, labels_test = prepare_labels(labels_train, labels_test)

    data_test, data_val, labels_test, labels_val = split_test_val(data_test, labels_test)

    model = create_model(image_size)
    model_history = compile_model(model, data_train, labels_train, data_val, labels_val)
    evaluate_model(model, data_test, labels_test)

    loss, accuracy, validation_loss, validation_accuracy = get_statistics(model_history)
    show_result_plot(loss, accuracy, validation_loss, validation_accuracy)

    if dataset_name == "svhn":
        save_model_tflite(model)


if __name__ == "__main__":
    main("svhn")
