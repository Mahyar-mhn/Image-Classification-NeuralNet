import numpy as np
import os
from os import listdir
from PIL import Image


def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max(x) for numerical stability
    return exp_x / np.sum(exp_x, axis=0)


def initialize_weights(input_size, hidden_size, output_size):
    w1 = np.random.uniform(-1, 1, (input_size, hidden_size))
    w2 = np.random.uniform(-1, 1, (hidden_size, output_size))
    b1 = np.random.random(hidden_size)
    b2 = np.random.random(output_size)
    return w1, w2, b1, b2


def process_image(image_path):
    img = Image.open(image_path).resize((5, 5)).convert('L')
    arrimage = np.asarray(img).flatten() / 255
    return arrimage


def train_neural_network(train_path, input_size, hidden_size, output_size):
    w1, w2, b1, b2 = initialize_weights(input_size, hidden_size, output_size)
    for image_name in listdir(train_path):
        image_path = os.path.join(train_path, image_name)
        arrimage = process_image(image_path)

        lay2 = arrimage.dot(w1) + b1
        l2 = softmax(lay2)

        lay3 = l2.dot(w2) + b2
        l3 = softmax(lay3)

        known_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'D', 'Gh', 'H', 'J', 'L', 'M', 'N',
                        'P', 'PuV', 'PwD', 'Sad', 'Sin', 'Ta', 'Taxi', 'V', 'Y']
        target_label = image_name[:3]
        target_index = known_labels.index(target_label)
        l3[target_index] -= 1

        javab = np.outer(l3, l2)

        b2 += l3
        b1 += l3.dot(w2.T)
        w2 -= 0.01 * javab
        w1 -= 0.01 * np.outer(arrimage, l2.dot(w2.T))

    return w1, w2, b1, b2


# Example usage:
train_data_path = "D:/discrete mathematic's project/phase 2/Train"
input_size = 25  # 5x5 image
hidden_size = 3
output_size = 28  # Number of classes
w1, w2, b1, b2 = train_neural_network(train_data_path, input_size, hidden_size, output_size)
