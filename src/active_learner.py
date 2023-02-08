import math
from datetime import datetime
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.autograd import Variable
from torch.optim import Adam
from helper_functions import log_message, annotate_img
import os

from custom_image_dataset import CustomImageDataset
from data_labelling_window import DataLabellingWindow
from examine_images_popup import ExamineImagesPopup

from supervisor_labeling_strategies.supervisor_labeling_strategy import SupervisorLabelingStrategy

import torchvision.transforms as T
from logging_level import LoggingLevel
import json


class ActiveLearner:

    def __init__(self, unlabeled_location: str, output_location: str, training_perc: int,
                 supervisor_labeling_strategy: SupervisorLabelingStrategy, neural_net: nn, neural_net_selection: str,
                 validation_perc: int, cross_validation: bool, classes: List[str], labeled_images_location: str,
                 query_size: int, num_epochs: int, batch_size: Optional[int] = 64):
        self._completed_epochs = 0
        self._num_epochs = num_epochs
        self._query_size = query_size
        self._batch_size = batch_size
        self._img_size = 224

        self._use_cross_validation = cross_validation
        self._class_list = classes
        self._annotation_filename = 'annotations.csv'

        self._loss_fn = nn.CrossEntropyLoss()
        self._neural_network: nn = neural_net
        self._neural_network_selection = neural_net_selection
        self._optimizer = Adam(self._neural_network.parameters(), lr=0.0001, weight_decay=0.0001)
        self._supervisor_labeling_strategy: SupervisorLabelingStrategy = supervisor_labeling_strategy

        self._training_perc = training_perc
        self._testing_perc = 100 - training_perc
        self._validation_perc = validation_perc

        self._output_location = output_location
        self._unlabeled_images_location = unlabeled_location
        self._labeled_images_location = labeled_images_location if labeled_images_location is not None \
            else os.path.abspath('./data/labeled')

        self._labeled_images = self._load_labeled_images()
        self._unlabeled_images = self._load_unlabeled_images()

        self._unlabeled_set = torch.utils.data.DataLoader(
            self._unlabeled_images, batch_size=self._batch_size, shuffle=True
        )

        log_message(f'Built active learner.{" Unlabeled images are in " if unlabeled_location != "" else ""}'
                    f'{os.path.basename(unlabeled_location)}'
                    f'{" Labeled images are in " if labeled_images_location != "" else ""}'
                    f'{os.path.basename(labeled_images_location)}. '
                    f' outputting model in {os.path.basename(output_location)}.\n\tTraining testing split: '
                    f'{training_perc}/{self._testing_perc}\n\tValidation training split: '
                    f'{validation_perc}/{100 - validation_perc}\n\t'
                    f'Using cross validation: {cross_validation}\n\tAll classes: {classes}',
                    LoggingLevel.INFO)
        log_message('Generating training and testing data...', LoggingLevel.DEBUG)
        self._test_set, self._training_set, self._validation_set = self._generate_test_train_validation()

        self._evaluated_unlabeled = None

        self._figure = None
        self._ax = None

        self._supervisor_labeling_idx = 0

    def save_model(self, model_name=f'model_{datetime.now().strftime("%H-%M-%S")}'):
        # Function to save the model
        output_location = os.path.join(self._output_location, model_name)
        if not os.path.exists(output_location):
            os.makedirs(output_location)
        metadata = {
            'model_type': self._neural_network_selection,
            'epochs_elapsed': self._completed_epochs,
            'classes': self._class_list
        }
        with open(os.path.join(output_location, 'metadata.json'), 'w') as file:
            json.dump(metadata, file)
        full_path = os.path.join(output_location, model_name)
        torch.save(self._neural_network.state_dict(), full_path)

    def _test_accuracy(self, dataset, return_full_results: Optional[bool] = False):
        # Function to test the model with the test dataset and print the accuracy for the test images
        model = self._neural_network
        model.eval()
        accuracy = 0.0
        total = 0.0
        results = []
        saved_image_paths = []
        features = None
        f_labels = None

        with torch.no_grad():
            for data in dataset:
                images, labels, paths = data
                images = images.float()

                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                images = Variable(images.to(device))
                labels = Variable(labels.to(device))

                saved_image_paths += list(paths)
                # run the model on the test set to predict labels
                model.to(device)
                outputs = model(images)
                # current_outputs = outputs.cpu().numpy()
                features = outputs.cpu() if features is None else np.concatenate((features, outputs.cpu()))
                # features = np.concatenate((features, cur_features)) if features is not None else cur_features
                f_labels = np.concatenate((f_labels, labels.cpu())) if f_labels is not None else labels.cpu()

                probabilities = nn.functional.softmax(outputs, dim=1)
                results += (list(zip(list(probabilities.cpu().numpy()), labels.cpu().numpy())))
                # print(results)
                # log_message(f'{soft.numpy()}', LoggingLevel.WARNING)
                # the label with the highest energy will be our prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                accuracy += (predicted == labels).sum().item()
        model.to('cpu')
        # compute the accuracy over all test images
        accuracy = (100 * accuracy / total)
        ret = (accuracy, results, (features, f_labels, saved_image_paths)) if return_full_results else accuracy

        return ret

    def train(self):
        """
        This function trains the model for the number of epochs given at instantiation. For each epoch we iterate over
        our image data training set and do backward propagation based on the given labels. At the end of each epoch, we
        check the validation score and save the
        """
        # First, make sure all the labeled and unlabeled images are loaded in correctly at the start of each training
        # session.
        self._labeled_images = self._load_labeled_images()
        self._unlabeled_images = self._load_unlabeled_images()

        self._evaluated_unlabeled = None
        self._supervisor_labeling_idx = 0

        log_message(f'Starting training over dataset in {self._num_epochs} epochs!')
        model = self._neural_network
        best_accuracy = 0.0

        # Define your execution device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log_message(f"The model will be running on {device} device", LoggingLevel.DEBUG)
        # Convert model parameters and buffers to CPU or Cuda
        model.to(device)

        for epoch in range(self._num_epochs):  # loop over the dataset multiple times
            running_loss = 0.0

            for i, (images, labels, paths) in enumerate(self._training_set, 0):
                # get the inputs
                images = images.float()
                images = Variable(images.to(device))
                labels = Variable(labels.to(device))

                # zero the parameter gradients
                self._optimizer.zero_grad()
                # predict classes using images from the training set
                images = images[:, :3, :, :]
                # print(images.shape)
                model.to(device)
                outputs = model(images)
                # current_outputs = outputs.cpu().numpy()
                # self._features = np.concatenate((outputs, current_outputs))
                # compute the loss based on model output and real labels
                loss = self._loss_fn(outputs, labels)
                # back-propagate the loss
                loss.backward()
                # adjust parameters based on the calculated gradients
                self._optimizer.step()

                running_loss += loss.item()  # extract the loss value
                if i % int(self._batch_size / 2) == int(self._batch_size / 2) - 1:
                    # print twice per epoch
                    log_message('[%d, %5d] loss: %.5f' %
                                (epoch + 1, i + 1, running_loss / self._batch_size),
                                LoggingLevel.DEBUG)
                    # zero the loss
                    running_loss = 0.0
            # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
            accuracy = self._test_accuracy(self._validation_set)
            self._completed_epochs += 1
            log_message(f'For epoch {self._completed_epochs} the accuracy over the validation set is '
                        f'{accuracy: .3f}%', LoggingLevel.INFO)

            # we want to save the model if the accuracy is the best
            if accuracy > best_accuracy:
                self.save_model()
                best_accuracy = accuracy

        log_message('Finished training!')
        accuracy = self._test_accuracy(self._test_set)
        log_message(f'Accuracy over the whole testing set is {accuracy: .3f}%', LoggingLevel.INFO)
        log_message(f'Model with best validation accuracy has been saved in {os.path.basename(self._output_location)}.')

    def supervisor_labeling(self, create_popup: Optional[bool] = True):
        """
        Queries the user or 'supervisor' for labels on a number of images equal to `self._query_size`. After getting
        every label, it places the images in the corresponding directories and regenerates the training data.
        """
        # Check to be sure there are some images to label...
        if not os.path.exists(self._unlabeled_images_location) or len(os.listdir(self._unlabeled_images_location)) == 0:
            log_message("No unlabeled images to label!", LoggingLevel.INFO)
        # Load the unlabeled images (needs to run each time this is called, as this method removes unlabeled images.)
        self._unlabeled_images = self._load_unlabeled_images()
        self._unlabeled_set = torch.utils.data.DataLoader(
            self._unlabeled_images, batch_size=self._batch_size, shuffle=True
        )
        # Get the results if they haven't already been found...
        accuracy, results, (features, f_labels, saved_image_paths) = self._test_accuracy(self._unlabeled_set, True) if \
            self._evaluated_unlabeled is None else self._evaluated_unlabeled
        # print(results)
        # Make sure the _evaluated_unlabeled field is updated...
        self._evaluated_unlabeled = (accuracy, results, (features, f_labels, saved_image_paths))
        # Use the desired Supervisor Labeling strategy to find the `n` most useful images to label...
        indices = self._supervisor_labeling_strategy.query_data(results=results)
        image_paths = [saved_image_paths[i] for i in indices]
        # We also have to remove all the duplicates in the list
        # image_paths = list(set(image_paths))
        if create_popup:
            # Create the pop-up gui for labelling the images, moving them into the correct location.
            labelling_window = DataLabellingWindow(self._class_list,
                                                   image_paths[self._supervisor_labeling_idx:
                                                               self._supervisor_labeling_idx + self._query_size],
                                                   self._labeled_images_location,
                                                   self._unlabeled_images_location)
            self._supervisor_labeling_idx += self._query_size
            labelling_window.run_gui()
            # Reload labeled images dataset with newly labeled images
            self._labeled_images = self._load_labeled_images()
            # Reload unlabeled images now that some have been removed...
            self._unlabeled_images = self._load_unlabeled_images()
            self._unlabeled_set = torch.utils.data.DataLoader(
                self._unlabeled_images, batch_size=self._batch_size, shuffle=True
            )
        return image_paths

    def machine_label(self, query_size_perc: Optional[float] = 0.1):
        # Get the paths to each image in order of least confident to most confident based on the given Supervisor
        # Labeling strategy.
        image_paths = self.supervisor_labeling(create_popup=False)
        # Reverse the list, so we get the most confident first...
        image_paths.reverse()
        # Remove all the images that we aren't confident enough about...
        image_paths = image_paths[:math.floor(len(image_paths) * query_size_perc)]
        popup = ExamineImagesPopup(self._class_list, image_paths, self._output_location,
                                   self._unlabeled_images_location, self._neural_network,
                                   T.Resize((self._img_size, self._img_size)))
        popup.run_gui()

    def _generate_test_train_validation(self):
        if self._labeled_images is None:
            log_message(f'No labeled images found. Querying {self._query_size} random images for you to label...')
            self.supervisor_labeling()
        total_count = int(len(self._labeled_images))
        train_count = int(total_count * (self._training_perc / 100))
        test_count = total_count - train_count
        valid_count = int(train_count * (self._validation_perc / 100))
        train_count -= valid_count
        training_set, validation_set, test_set = torch.utils.data.random_split(
            self._labeled_images, (train_count, valid_count, test_count)
        )

        log_message('Split labeled data into testing, training and validation...', LoggingLevel.DEBUG)
        log_message(f'{"Not u" if self._use_cross_validation is False else "U"}sing cross validation to generate the '
                    f'validation set.', LoggingLevel.DEBUG)
        if self._use_cross_validation:
            log_message('Cross validation has not been implemented yet.', LoggingLevel.WARNING)

        train_dataset_loader = torch.utils.data.DataLoader(
            training_set, batch_size=self._batch_size, shuffle=True
        )
        test_dataset_loader = torch.utils.data.DataLoader(
            test_set, batch_size=self._batch_size, shuffle=False
        )
        validation_dataset_loader = torch.utils.data.DataLoader(
            validation_set, batch_size=self._batch_size, shuffle=True
        )

        return test_dataset_loader, train_dataset_loader, validation_dataset_loader

    def _load_labeled_images(self) -> CustomImageDataset:
        labeled_images = None
        if self._labeled_images_location != '':
            annotations_fp = os.path.join(self._labeled_images_location, self._annotation_filename)
            if os.path.exists(annotations_fp):
                os.remove(annotations_fp)
            for i, nn_class in enumerate(self._class_list):
                if i == 0:
                    log_message('Finding labeled images...', LoggingLevel.DEBUG)
                path = os.path.join(self._labeled_images_location, nn_class)
                if os.path.exists(path):
                    log_message(f'Found labeled images for class "{nn_class}"!', LoggingLevel.DEBUG)
                    for filepath in os.listdir(path):
                        annotate_img(annotations_fp,
                                     (os.path.join(nn_class, os.path.basename(filepath))),
                                     i)
            labeled_images = CustomImageDataset(annotations_fp, self._labeled_images_location,
                                                transform=T.Resize((self._img_size, self._img_size)))

        return labeled_images

    def _load_unlabeled_images(self) -> CustomImageDataset:
        unlabeled_images = None
        if self._unlabeled_images_location != '':
            log_message(f'Finding unlabeled images...', LoggingLevel.DEBUG)
            annotations_fp = os.path.join(self._unlabeled_images_location, self._annotation_filename)
            if os.path.exists(annotations_fp):
                os.remove(annotations_fp)
            for filepath in os.listdir(self._unlabeled_images_location):
                annotate_img(annotations_fp, os.path.basename(filepath), -1)
            unlabeled_images = CustomImageDataset(annotations_fp, self._unlabeled_images_location,
                                                  transform=T.Resize((self._img_size, self._img_size)))

        return unlabeled_images

    @staticmethod
    def _scale_to_01_range(x):
        value_range = (np.max(x) - np.min(x))
        starts_from_zero = x - np.min(x)
        return starts_from_zero / value_range

    def get_tsne(self) -> plt.Figure:
        if self._figure is None:
            figure, ax = plt.subplots(figsize=(10, 10), dpi=100)
            self._figure = figure
            self._ax = ax
        self._ax.cla()
        t_accuracy, t_results, (t_features, tf_labels, saved_image_paths) = self._test_accuracy(self._test_set, True)
        u_accuracy, u_results, (u_features, uf_labels, saved_image_paths) = self._test_accuracy(self._unlabeled_set,
                                                                                                True) \
            if self._evaluated_unlabeled is None else self._evaluated_unlabeled
        # Make sure the _evaluated_unlabeled field is updated...
        self._evaluated_unlabeled = (u_accuracy, u_results, (u_features, uf_labels, saved_image_paths))
        features = np.concatenate((t_features, u_features))

        tsne = TSNE(n_components=2, init='pca', learning_rate='auto').fit_transform(features)
        tx = tsne[:, 0]
        ty = tsne[:, 1]

        labels = np.concatenate((tf_labels, uf_labels))
        tx = self._scale_to_01_range(tx)
        ty = self._scale_to_01_range(ty)

        indices = np.where(labels == -1)
        self._ax.scatter(tx[indices], ty[indices], label='unlabeled', alpha=0.6)
        for class_ind, class_name in enumerate(self._class_list):
            indices = [i for i, la in enumerate(labels) if la == class_ind]
            cur_tx = np.take(tx, indices)
            cur_ty = np.take(ty, indices)
            self._ax.scatter(cur_tx, cur_ty, label=class_name)

        self._ax.legend()
        self._figure.canvas.draw()
        return self._figure

    def update_query_size(self, query_size):
        self._query_size = query_size
