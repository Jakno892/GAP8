# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""model training for light source recognition and distance estimation.

Run:
`bazel build tensorflow/lite/micro/examples/hello_world:train`
`bazel-bin/tensorflow/lite/micro/examples/hello_world/train --save_tf_model --save_dir=/tmp/model_created/`
"""
import math
import os
from PIL import Image

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
import cv2

FLAGS = flags.FLAGS

flags.DEFINE_integer("epochs", 500, "number of epochs to train the model.")
flags.DEFINE_string("save_dir", "/examensarbete/AI-deck/image_processing/models",
                    "the directory to save the trained model.")
flags.DEFINE_boolean("save_tf_model", False,
                     "store the original unconverted tf model.")


def get_data():
    """
    The code will read from a file containing the expected network outputs.
    """
    # Open datafile and read from each line.

    on_off = []
    dist_est = []
    pix_x= []
    pix_y = []

    data = open("~/examensarbete/AI-deck/image_processing/data_collection/data.txt","r")
    lines = data.readlines()
    for line in lines:
        split_line = line.split(",")
        on_off.append(split_line[0])
        dist_est.append(float(split_line[1])*int(split_line[0]))
        pix_x.append(int(split_line[2]))
        pix_y.append(int(split_line[3].replace("\n", "")))
    index = 1
    images = []
    while index < 644:
        img = Image.open("~/examensarbete/AI-deck/image_processing/images/recording_" + str(index) + ".png").convert('LA')
        images.append(img)

    labels = [on_off, dist_est, pix_x, pix_y]
    return (images, labels)


def create_model() -> tf.keras.Model:
    model = tf.keras.Sequential()

    # First layer takes a scalar input and feeds it through 16 "neurons". The
    # neurons decide whether to activate based on the 'relu' activation function.
    model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(320,320,1)))

    # The new second and third layer will help the network learn more complex
    # representations
    model.add(tf.keras.layers.Dense(16, activation='relu'))

    # Final layer is four neurons, since we want to output:
    # - Wether the optical source is in the frame (Bool 1/0)
    # - The estimated distance to the optical source (Float)
    # - The pixel coordinates of the optical source (int, int).
    model.add(tf.keras.layers.Dense(4))

    # Compile the model using the standard 'adam' optimizer and the mean squared
    # error or 'mse' loss function for regression.
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model


def convert_tflite_model(model):
    """Convert the save TF model to tflite model, then save it as .tflite flatbuffer format
        Args:
            model (tf.keras.Model): the trained hello_world Model
        Returns:
            The converted model in serialized format.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    return tflite_model


def save_tflite_model(tflite_model, save_dir, model_name):
    """save the converted tflite model
    Args:
      tflite_model (binary): the converted model in serialized format.
      save_dir (str): the save directory
      model_name (str): model name to be saved
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, model_name)
    with open(save_path, "wb") as f:
        f.write(tflite_model)
    logging.info("Tflite model saved to %s", save_dir)


def train_model(epochs, input_data, ground_truth_data):
    """Train keras hello_world model
    Args: epochs (int) : number of epochs to train the model
        x_train (numpy.array): list of the training data
        y_train (numpy.array): list of the corresponding array
    Returns:
        tf.keras.Model: A trained keras hello_world model
    """
    model = create_model()
    model.fit(input_data,
            ground_truth_data,
            epochs=epochs,
            validation_split=0.2,
            batch_size=64,
            verbose=2)

    if FLAGS.save_tf_model:
        model.save(FLAGS.save_dir, save_format="tf")
        logging.info("TF model saved to %s", FLAGS.save_dir)

    return model


def main(_):
    input_data, ground_truth_data = get_data()
    trained_model = train_model(FLAGS.epochs, input_data, ground_truth_data)

    # Convert and save the model to .tflite
    tflite_model = convert_tflite_model(trained_model)
    save_tflite_model(tflite_model,
                    FLAGS.save_dir,
                    model_name="distance_estimation.tflite")


if __name__ == "__main__":
  app.run(main)