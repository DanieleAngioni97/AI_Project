from prova_lettura_immagini import read_pgm_reshape
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import os


def create_csv():
    images_path = []
    labels = []
    for (dirpath, dirnames, filenames) in os.walk(top=r"DaimlerBenchmark"):
        if not dirnames:
            for filename in filenames:
                images_path.append(os.path.join(dirpath, filename))
                labels.append(0 if "non-ped" in dirpath else 1)

    data = np.array([images_path, labels]).T
    random.shuffle(data)
    df = pd.DataFrame(data, columns=['filename', 'label'])
    df.to_csv("DaimlerBenchmark/pedestrian_dataset.csv", index=False)

    df1 = pd.read_csv("DaimlerBenchmark/pedestrian_dataset.csv")
    print("")



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
    random.shuffle(train)   # todo: controllare che abbia mischiato le righe e non gli elementi a caso

    for i, el in enumerate(train):
        x_tr[i] = el[0]
        y_tr[i] = el[1]

    return x_tr, y_tr

if __name__ == "__main__":
    create_csv()