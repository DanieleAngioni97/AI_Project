from prova_lettura_immagini import read_pgm_reshape
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import os

def load_pedestrian_dataset():
    os.chdir(r"C:\Users\angio\PycharmProjects\AI_Project\DaimlerBenchmark\1")
    path1 = os.getcwd()
    non_ped_path = path1 + r"\non-ped_examples\\"
    ped_path = path1 + r"\ped_examples\\"

    x_tr = []
    y_tr = []
    for folder_path in [(non_ped_path, 0), (ped_path, 1)]:
        for img_path in os.listdir(folder_path[0])[:50]:
            img = read_pgm_reshape(filename=folder_path[0] + img_path, new_shape=(32, 32))
            img = np.asarray(img)
            x_tr.append(img)
            y_tr.append(folder_path[1])

    x_tr, y_tr = np.array(x_tr).astype('float32') / 255., np.array(y_tr)
    train = list(zip(x_tr, y_tr))
    random.shuffle(train)

    for i, el in enumerate(train):
        x_tr[i] = el[0]
        y_tr[i] = el[1]

    return x_tr, y_tr


def plot_images_from_dataset(data):
    n = 5  # images to be visualized
    bias = 0  # starting index from the test set for the visualization
    plt.figure(figsize=(20, 10))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(data[i + bias], plt.cm.gray)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

