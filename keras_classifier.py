import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import ssl

from audio_processing import get_audio_array_process
from utils import load_obj

ssl._create_default_https_context = ssl._create_unverified_context

dataset = load_obj('entries.pickle')
min_len = 10000
max_len = 1
min_v = 10000
max_v = -10000

instances = []
labels = []

for entry in dataset:
    (instance, label) = entry
    instance = instance + 5000
    instance = instance / 10000
    instances.append(instance[:42])
    labels.append(np.argmax(label))
    # max_len = max(max_len, len(instance))
    # min_len = min(min_len, len(instance))
    # for v in instance:
    #     max_v = max(max_v, v[0])
    #     max_v = max(max_v, v[1])
    #     min_v = min(min_v, v[0])
    #     min_v = min(min_v, v[1])

instances = np.array(instances)
labels = np.array(labels)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = instances
train_labels = labels
test_images = instances
test_labels = labels

class_names = ['t', 'v', 'c']

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(42, 2)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(100, activation=tf.nn.relu),
    keras.layers.Dense(40, activation=tf.nn.relu),
    keras.layers.Dense(3, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

def train():

    model.fit(train_images, train_labels, epochs=50)

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print('Test accuracy:', test_acc)

    model.save_weights('data/')


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)
    # plt.show()


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
    # plt.show()


def predict_second(sound_array, length_secs, second_index):
    start = int(second_index / length_secs * len(sound_array))
    end = int((second_index + 1) / length_secs * len(sound_array))
    sound = sound_array[start:start+42]
    if len(sound) != 42:
        return
    sound = (np.expand_dims(sound, 0))
    prediction = model.predict(sound)[0]
    return prediction


def predict_all():
    # model.load_weights('data/checkpoint')
    sound = get_audio_array_process('videos/Podcast 216  Sunt podcasturile degeaba  Intre showuri cu Teo Vio si Costel.mp4')
    l = 54 * 50 + 5
    p = 'output.txt'
    with open(p, 'w') as handle:
        for s in range(l):
            p = predict_second(sound, l, s)
            min = s // 60
            sec = s % 60
            print('{}:{} -> {} Teo, {} Vio, {} Costel'.format(min, sec, p[0], p[1], p[2]))


train()
predict_all()


# predictions = model.predict(test_images)
#
# img = test_images[0]
# img = (np.expand_dims(img, 0))
# predictions_single = model.predict(img)
# plot_value_array(0, predictions_single, test_labels)
# _ = plt.xticks(range(10), class_names, rotation=45)
# print(class_names[np.argmax(predictions_single[0])])
