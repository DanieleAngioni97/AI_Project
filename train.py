import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from model import ConvNet
from pedestrian_dataset import PedestrianDataset
import utils

# Hyper-parameters

batch_size = 64
validation_split = 0.2
n_iteration = 5
num_epochs_new = 10
learning_rate = 0.001

model_with_augmentation = "ConvNet_with_augmentation"
model_without_augmentation = "ConvNet_without_augmentation"
model_names_list = [model_with_augmentation, model_without_augmentation]
save = False
pretrained = False

utils.set_seed(random_seed=49)

transform_without_augmentation = transforms.ToTensor()
transform_with_augmentation = transforms.Compose([
        transforms.ColorJitter(brightness=(0.8, 1.2),
                               contrast=(0.8, 1.2),
                               saturation=(0.8, 1.2)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6)),
        transforms.ToTensor(),
        ])
transform_list = [transform_without_augmentation, transform_with_augmentation]

# set CPU or GPU, if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_orig = ConvNet().to(device)  # instantiate the model

for transform, model_name in list(zip(transform_list, model_names_list)):
    # in the first iteration will be trained the ConvNet without augmentation
    # and in the second iteration the ConvNet with augmentation

    # create the dataset that contains the test set and the validation set and apply the transformation
    dataset = PedestrianDataset(train=True,
                                transform=transform)


    model = ConvNet().to(device)
    model.load_state_dict(model_orig.state_dict())  # in this way I'm sure that both the ConvNets start
    # with the same weight

    # if pretrained is True, load the weight and the other data from the old model
    if pretrained:
        checkpoint = torch.load(utils.MODELS_PATH + model_name + ".tar", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        (tr_loss_path_old, val_loss_path_old) = checkpoint['loss']
        batch_size = checkpoint['batch_size']
        validation_split = checkpoint['validation_split']
        total_step_old = checkpoint['total_step']
        n_iteration = checkpoint['n_iteration']
        num_epochs_old = checkpoint['epoch']

    # create dataloaders (see pedestrian_dataset.py)
    train_loader, validation_loader = dataset.loader(batch_size=batch_size,
                                                     validation_split=validation_split,
                                                     shuffle_dataset=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    tr_loss_path = np.zeros(shape=(num_epochs_new, int(total_step/n_iteration)))
    val_loss_path = np.zeros(shape=(num_epochs_new, int(total_step/n_iteration)))

    # temporary variable to save n_iteration loss for the train
    temp_tr_loss = np.zeros(shape=n_iteration)

    # if pretrained is True, load the old losses in the vector and add the old number of epochs in num_epochs
    if pretrained:
        tr_loss_path = np.vstack([tr_loss_path_old, tr_loss_path])
        val_loss_path = np.vstack([val_loss_path_old, val_loss_path])
        num_epochs = num_epochs_old + 1 + num_epochs_new
    else:
        # if pretrained is False the total number of epochs is equal to the number choose in the top of the code
        num_epochs = num_epochs_new

    # Train the model
    for epoch in range(num_epochs_new):
        if pretrained:
            # if pretrained is True, add to the epoch index the old number of epochs already done
            epoch = num_epochs_old + 1 + epoch

        # n is the number of iteration
        n = 0   # n is the number of iterations per n_iteration (exactly the same size as the graph)
        k = 0   # k is a temporary index ranging from 0 to n_iteration and is used to fill temp_tr_loss
        # i is the index of the batch

        for i, (tr_images, tr_labels) in enumerate(train_loader):
            # pass the current image and the true label to the device
            tr_images = tr_images.to(device)
            tr_labels = tr_labels.to(device)

            # Forward pass
            tr_outputs = model(tr_images)[-1]
            tr_loss = criterion(tr_outputs, tr_labels)
            temp_tr_loss[k] = tr_loss.item()

            # Set all the local gradients to zero
            optimizer.zero_grad()

            # Backpropagate the error
            tr_loss.backward()

            # Update parameters
            optimizer.step()

            k = k + 1
            if (i+1) % n_iteration == 0:
                # each time n_iteration are done reinitialise the index k to 0 and makes the mean of temp_tr_loss
                # and evaluate the validation loss
                k = 0
                temp_val_loss = np.zeros(shape=(len(validation_loader)))
                model.eval()  # eval mode (batchnorm uses moving mean/var instead of mini-batch mean/var)

                with torch.no_grad():  # for the validation is not necessary to calculate the gradients
                    for j, (val_images, val_labels) in enumerate(validation_loader):
                        # pass the current image and the true label to the device
                        val_images = val_images.to(device)
                        val_labels = val_labels.to(device)

                        # Forward pass
                        val_outputs = model(val_images)[-1]
                        val_loss = criterion(val_outputs, val_labels)
                        temp_val_loss[j] = val_loss.item()

                model.train()  # set the model in training mode

                # compute the mean for the train and validation loss and save the minimum for the validation loss
                tot_val_loss = temp_val_loss.mean()
                tr_loss_path[epoch][n] = temp_tr_loss.mean()
                val_loss_path[epoch][n] = tot_val_loss

                if n == 0:
                    min_val_loss = tot_val_loss
                # if a smaller loss is reached save the model and the various data
                if tot_val_loss < min_val_loss:
                    min_val_loss = tot_val_loss

                    if save:
                        torch.save({
                            'model_name': model_name,
                            'epoch': epoch,
                            'num_epochs': num_epochs,
                            'batch_size': batch_size,
                            'validation_split': validation_split,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': (tr_loss_path, val_loss_path),
                            'n_iteration': n_iteration,
                            'total_step': total_step
                        }, utils.MODELS_PATH + model_name + ".tar")

                print('Epoch [{}/{}], Step [{}/{}], Train loss: {:.4f}, Validation loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step,
                              tr_loss_path[epoch][n], val_loss_path[epoch][n]))
                n = n + 1
            if (i+1) % 100 == 0:
                torch.cuda.empty_cache()
