import csv
import datetime
import os
from typing import Iterable, Optional

import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from custom_image_dataset import CustomImageDataset
from image_neural_network import Network


class ModelUtils:

    def __init__(self, model):
        self._model = model
        # self._classes = model.classes
        self._loss_fn = nn.CrossEntropyLoss()
        self._optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
        self._training_data = None
        self._validation_data = None
        self._unlabeled_data_location = os.path.abspath('./unlabeled')
        self._batch_size = 64
        # self._writer = SummaryWriter('runs/bee_parts')

    def save_model(self, path='./models', model_name=f'model_{datetime.datetime.now()}'):
        # Function to save the model
        if not os.path.exists(path):
            os.makedirs(path)
        full_path = os.path.join(path, model_name)
        torch.save(self._model.state_dict(), full_path)

    def test_accuracy(self):
        # Function to test the model with the test dataset and print the accuracy for the test images
        model = self._model
        model.eval()
        accuracy = 0.0
        total = 0.0

        with torch.no_grad():
            for data in self.validation_loader():
                images, labels, paths = data
                images = images.float()
                # run the model on the test set to predict labels
                outputs = model(images)
                # print(tuple(torch.max(outputs.data, 1)))
                # the label with the highest energy will be our prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                accuracy += (predicted == labels).sum().item()

        # compute the accuracy over all test images
        accuracy = (100 * accuracy / total)
        return accuracy

    def return_least_confident(self):
        """
        Returns a list of filepaths for every unlabeled image in the order of least to most confident.
        """
        self._model.eval()

        with torch.no_grad():
            filename_list, data_loader = self.unlabeled_test_loader()
            conf_path = []
            for data in data_loader:
                images, labels, paths = data
                images = images.float()
                # run the model on the test set to predict labels
                outputs = self._model(images)
                out = tuple(torch.max(outputs.data, 1))
                conf_path.append((out[0].numpy()[0], paths[0], out[1].numpy()[0]))
                # print(filename_list[i][1])
                # print(filename_list[i])
            conf_path.sort(key=lambda a: a[0])
            for j in range(len(conf_path) - 1, len(conf_path) - 11, -1):
                print(conf_path[j])
            return [os.path.basename(item[1]) for item in conf_path]

    def train(self, num_epochs):
        """
        Training function. We simply have to loop over our data iterator and feed the inputs to the network and
        optimize.
        """
        model = self._model
        best_accuracy = 0.0

        # Define your execution device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("The model will be running on", device, "device")
        # Convert model parameters and buffers to CPU or Cuda
        model.to(device)

        for epoch in range(num_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            running_acc = 0.0

            for i, (images, labels, paths) in enumerate(self.train_loader(), 0):
                # get the inputs
                images = images.float()
                images = Variable(images.to(device))

                labels = Variable(labels.to(device))

                # zero the parameter gradients
                self._optimizer.zero_grad()
                # predict classes using images from the training set
                outputs = model(images)
                # compute the loss based on model output and real labels
                loss = self._loss_fn(outputs, labels)
                # back-propagate the loss
                loss.backward()
                # adjust parameters based on the calculated gradients
                self._optimizer.step()

                # Let's print statistics for every 1,000 images
                running_loss += loss.item()  # extract the loss value
                if i % int(self._batch_size / 2) == int(self._batch_size / 2) - 1:
                    # print twice per epoch
                    print('[%d, %5d] loss: %.5f' %
                          (epoch + 1, i + 1, running_loss / self._batch_size))
                    # zero the loss
                    running_loss = 0.0

            # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
            accuracy = self.test_accuracy()
            print('For epoch', epoch + 1, 'the validation accuracy over the whole validation set is %d %%' % accuracy)

            # we want to save the model if the accuracy is the best
            if accuracy > best_accuracy:
                self.save_model()
                best_accuracy = accuracy

    def test_batch(self, batch_size=1000):
        # Function to test the model with a batch of images and show the labels predictions
        classes = self._classes
        # get batch of images from the test DataLoader
        images, labels, paths = next(iter(self.validation_loader()))

        # show all images as one image grid
        self.image_show(torchvision.utils.make_grid(images))

        # Show the real labels on the screen
        print('Real labels: ', ' '.join('%5s' % classes[labels[j]]
                                        for j in range(batch_size)))

        # Let's see what if the model identifiers the  labels of those example
        outputs = self._model(images)

        # We got the probability for every 10 labels. The highest (max) probability should be correct label
        _, predicted = torch.max(outputs, 1)

        # Let's show the predicted labels on the screen to compare with the real ones
        print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                      for j in range(batch_size)))

    def train_validation_loader(self, percent_train: Optional[float] = 0.7, shuffle=True):
        """

        Args:
            percent_train:
            shuffle:
        Returns:
             Iterable:
        """
        idx = 0
        train_rows = []
        validation_rows = []
        total_rows = []
        with open(os.path.join('./data', 'annotations.csv'), 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                total_rows.append(row)
            row_count = len(total_rows)
            for row in total_rows:
                if idx < row_count * percent_train:
                    train_rows.append(row)
                else:
                    validation_rows.append(row)
                idx += 1
        with open(os.path.join('./data', 'annotations.csv'), 'r') as csvfile:
            reader = csv.reader(csvfile)
            self._resize_data(reader, 227, 227)

        self._csv_write('./data/training/', 'annotations.csv', train_rows)
        self._csv_write('data/validation/', 'annotations.csv', validation_rows)
        annotations_path = os.path.join('./data/training', 'annotations.csv')
        training_data = CustomImageDataset(annotations_path, './data')
        training_data = DataLoader(training_data, batch_size=self._batch_size, shuffle=shuffle)
        self._training_data = training_data

        annotations_path = os.path.join('data/validation', 'annotations.csv')
        validation_data = CustomImageDataset(annotations_path, './data')
        validation_data = DataLoader(validation_data, batch_size=self._batch_size, shuffle=shuffle)
        self._validation_data = validation_data

    def validation_loader(self):
        if self._validation_data is None:
            self.train_validation_loader()
        return self._validation_data

    def train_loader(self):
        if self._training_data is None:
            self.train_validation_loader()
        elif os.path.exists(os.path.abspath('./data/new_training_data')):
            # Train loader has the special case of checking whether there are any images in the 'new_training_data'
            # directory and adding them to the current training data set.
            new_images_and_labels = []
            with open(os.path.join(os.path.abspath('./data'), 'annotations.csv'), 'r') as csvfile:
                reader = csv.reader(csvfile)
                new_training_images = [os.path.join(os.path.abspath('./data/new_training_data'), file)
                                       for file in os.listdir(os.path.abspath('./data/new_training_data'))]
                corrected_image_paths = [os.path.join(os.path.abspath('./data'), os.path.basename(name))
                                         for name in new_training_images]
                # print('\n\n')
                # print(corrected_image_paths)
                # print('\n\n')
                # print(new_training_images)
                # print('\n\n')
                for row in reader:
                    for i in range(len(corrected_image_paths)):
                        if row[0] == os.path.basename(corrected_image_paths[i]):
                            # print(f'renaming {new_training_images[i]} as {corrected_image_paths[i]}')
                            os.rename(os.path.join(
                                os.path.abspath('./data/new_training_images'), new_training_images[i]),
                                corrected_image_paths[i])
                            new_images_and_labels.append((os.path.basename(corrected_image_paths[i]), row[1]))
                            break
                self._csv_append(os.path.abspath('./data/training'), 'annotations.csv', new_images_and_labels)
            annotations_path = os.path.join('./data/training', 'annotations.csv')
            training_data = CustomImageDataset(annotations_path, './data')
            training_data = DataLoader(training_data, batch_size=self._batch_size, shuffle=False)
            self._training_data = training_data
        return self._training_data

    def unlabeled_test_loader(self):
        filename_list = os.listdir(self._unlabeled_data_location)
        unlabeled_rows = []
        for filename in filename_list:
            if filename[-3:] != 'csv':
                unlabeled_rows.append((filename, -1))
        self._csv_write(self._unlabeled_data_location, 'annotations.csv', unlabeled_rows)
        self._resize_unlabeled(filename_list, 227, 227)
        unlabeled_data = CustomImageDataset(os.path.join(self._unlabeled_data_location, 'annotations.csv'),
                                            self._unlabeled_data_location)
        unlabeled_data = DataLoader(unlabeled_data, batch_size=self._batch_size, shuffle=False)
        return filename_list, unlabeled_data

    def _resize_unlabeled(self, filenames, height, width):
        for filename in filenames:
            if filename == 'annotations.csv':
                continue
            img = cv2.imread(os.path.join(self._unlabeled_data_location, filename))
            try:
                img = cv2.resize(img, (width, height))
                cv2.imwrite(os.path.join(self._unlabeled_data_location, filename), img)
            except Exception as e:
                print(f'filename `{filename}` not in the unlabeled folder, but is in the annotations. Found through\n'
                      f'error {e}')

    @staticmethod
    def image_show(img):
        """
        Args:
            img:
        """
        img = img / 2 + 0.5  # denormalize
        np_img = img.numpy()
        plt.imshow(np.transpose(np_img, (1, 2, 0)))
        plt.show()

    @staticmethod
    def _csv_write(filepath, filename, data):
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        with open(os.path.join(filepath, filename), 'w') as new_file:
            writer = csv.writer(new_file)
            first = True
            for entry in data:
                if first:
                    writer.writerow(['path', 'class'])
                    first = False
                writer.writerow(entry)

    @staticmethod
    def _csv_append(filepath, filename, data):
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        with open(os.path.join(filepath, filename), 'a') as new_file:
            writer = csv.writer(new_file)
            writer.writerows(data)

    @staticmethod
    def _resize_data(csv_reader, height, width):
        for row in csv_reader:
            img = cv2.imread(os.path.join('./data', row[0]))
            img = cv2.resize(img, (width, height))
            cv2.imwrite(os.path.join('./data', row[0]), img)


if __name__ == '__main__':
    # Instantiate a neural network model
    classes_list = sys.argv[1]
    classes_list = classes_list.replace(',', '').split(' ')
    model_inst = Network(classes_list)
    model_utils = ModelUtils(model_inst)
    model_utils.train(5)
    print('finished training')
    # print(model_utils.test_accuracy())
    model_utils.return_least_confident()
