import pandas as pd
import numpy as np
from PIL import Image
import random
import torchvision.transforms as transforms
import os
import torch
import matplotlib.pyplot as plt

# CONSTANT

MODELS_PATH = "saved_models/"
ROOT_DIR_DATASET = "data/INRIA_cropped"
TRAIN_CSV_FNAME = "pedestrian_trainset.csv"
TEST_CSV_FNAME = "pedestrian_testset.csv"


def set_seed(random_seed=0):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def create_csv():
    """
    Explore the dataset and create two .csv, one for train and one for test
    """
    tr_images_path = []
    tr_labels = []
    tr_fname = "data/INRIA_cropped/pedestrian_trainset.csv"

    ts_images_path = []
    ts_labels = []
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


# USED FOR FIRST DATASET IN GREYSCALE
# def load_pedestrian_dataset():
#     os.chdir(r"C:\Users\angio\PycharmProjects\AI_Project\DaimlerBenchmark\1")
#     path1 = os.getcwd()
#     non_ped_path = path1 + r"\non-ped_examples\\"
#     ped_path = path1 + r"\ped_examples\\"
#
#     x_tr = []
#     y_tr = []
#     for folder_path in [(non_ped_path, 0), (ped_path, 1)]:
#         for img_path in os.listdir(folder_path[0])[:50]:
#             img = read_pgm_reshape(filename=folder_path[0] + img_path, new_shape=(32, 32))
#             img = np.asarray(img)
#             x_tr.append(img)
#             y_tr.append(folder_path[1])
#
#     x_tr, y_tr = np.array(x_tr).astype('float32') / 255., np.array(y_tr)
#     train = list(zip(x_tr, y_tr))
#     random.shuffle(train)
#
#     for i, el in enumerate(train):
#         x_tr[i] = el[0]
#         y_tr[i] = el[1]
#
#     return x_tr, y_tr

# def plot_images_from_dataset(data):
#     n = 5  # images to be visualized
#     bias = 0  # starting index from the test set for the visualization
#     plt.figure(figsize=(20, 10))
#     for i in range(n):
#         ax = plt.subplot(1, n, i + 1)
#         plt.imshow(data[i + bias])
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#
#     plt.show()

def statistics(main_path):
    """
    Convenience function to see dimensions of images in the dataset
    """
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
    """
    Creates 4 different list with the path from content root of all sample files in the dataset,
    divided in Test pedestrian, Test non-pedestrian, Train pedestrian, Train non-pedestrian
    """
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

    # take the list of all filenames of the sample
    test_pos, test_neg, train_pos, train_neg = walk_folders(main_path)

    rand_crop = transforms.RandomCrop(size=(128, 64))
    centr_crop = transforms.CenterCrop(size=(128, 64))

    for k, folder in enumerate((test_pos, train_pos, test_neg, train_neg)):
        for img_path in folder:
            img = Image.open(img_path)
            if '.jpg' in img_path:
                img_path = img_path.replace('.jpg', '.png')
            # if pedestrian do a central crop and save image in the destination folder
            if k < 2:
                new_img = centr_crop(img)
                new_path = img_path.replace("organized", "cropped")
                new_img.save(new_path)
            # if non-pedestrian do two random crop and save the two different crops in the destination folder
            if k >= 2:
                for i in range(2):
                    new_img = rand_crop(img)
                    new_path = img_path.replace("organized", "cropped").replace(".png", "_{}.png".format(i))
                    new_img.save(new_path)


if __name__ == "__main__":
    # create_new_dataset()
    # create_csv()
    statistics('data/INRIA_organized')


