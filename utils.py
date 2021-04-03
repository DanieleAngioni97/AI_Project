from read_pgm import read_pgm_reshape
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import torchvision.transforms as transforms
import os

# CONSTANT

PATH = "saved_models/"
ROOT_DIR_DATASET = "data/INRIA_cropped"
TRAIN_PATH = "pedestrian_trainset.csv"
TEST_PATH = "pedestrian_testset.csv"

def create_csv():
    ts_images_path = []
    ts_labels = []
    tr_fname = "data/INRIA_cropped/pedestrian_trainset.csv"

    tr_images_path = []
    tr_labels = []
    ts_fname = "data/INRIA_cropped/pedestrian_testset.csv"

    for (dirpath, dirnames, filenames) in os.walk(top="data/INRIA_cropped"):
        if not dirnames:
            for filename in filenames:
                if "Test" in dirpath:
                    ts_images_path.append(os.path.join(dirpath, filename))
                    if "pos" in dirpath:
                        ts_labels.append(1)
                    if "neg" in dirpath:
                        ts_labels.append(0)
                if "Train" in dirpath:
                    tr_images_path.append(os.path.join(dirpath, filename))
                    if "pos" in dirpath:
                        tr_labels.append(1)
                    if "neg" in dirpath:
                        tr_labels.append(0)

    for (images_path, labels, fname) in [(tr_images_path, tr_labels, tr_fname), (ts_images_path, ts_labels, ts_fname)]:
        data = np.array([images_path, labels]).T.tolist()
        data = np.array(data)
        df = pd.DataFrame(data, columns=['filename', 'label'])
        df.to_csv(fname, index=False)

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

def statistics(main_path):
    test_pos, test_neg, train_pos, train_neg = walk_folders(main_path)

    test_pos_shapes = []
    test_neg_shapes = []
    train_pos_shapes = []
    train_neg_shapes = []
    folders = [test_pos, test_neg, train_pos, train_neg]
    shapes = [test_pos_shapes, test_neg_shapes, train_pos_shapes, train_neg_shapes]

    for (folder, shape) in list(zip(folders, shapes)):
        for path in folder:
            image = np.asarray(Image.open(path))
            shape.append(image.shape)

    for shape in shapes:
        print(np.unique([s[0] for s in shape], return_counts=True))
        print(np.unique([s[1] for s in shape], return_counts=True))
        print("n items = ", len(shape))
        print("\n")

def walk_folders(main_path):
    test_pos = []
    test_neg = []
    train_pos = []
    train_neg = []

    for (dirpath, dirnames, filenames) in os.walk(top=main_path):
        if not dirnames:
            for filename in filenames:
                if 'Test' in dirpath:
                    if 'pos' in dirpath:
                        test_pos.append(os.path.join(dirpath, filename))
                    if 'neg' in dirpath:
                        test_neg.append(os.path.join(dirpath, filename))

                if 'Train' in dirpath:
                    if 'pos' in dirpath:
                        train_pos.append(os.path.join(dirpath, filename))
                    if 'neg' in dirpath:
                        train_neg.append(os.path.join(dirpath, filename))
    return test_pos, test_neg, train_pos, train_neg

def create_new_dataset():
    main_path = 'data/INRIA_organized'
    test_pos, test_neg, train_pos, train_neg = walk_folders(main_path)

    rand_crop = transforms.RandomCrop(size=(128, 64))
    centr_crop = transforms.CenterCrop(size=(128, 64))
    folders = [[], [], [], []]

    for k, folder in enumerate((test_pos, train_pos, test_neg, train_neg)):
        for img_path in folder:
            img = Image.open(img_path)
            if '.jpg' in img_path:
                img_path = img_path.replace('.jpg', '.png')
            if k < 2:
                new_img = centr_crop(img)
                new_path = img_path.replace("organized", "cropped")
                new_img.save(new_path)
            if k >= 2:
                for i in range(2):
                    new_img = rand_crop(img)
                    new_path = img_path.replace("organized", "cropped").replace(".png", "_{}.png".format(i))
                    folders[2].append(new_path)
                    new_img.save(new_path)
            print(new_path)



if __name__ == "__main__":
    # statistics("data/INRIA_cropped")
    create_csv()



    print("")


