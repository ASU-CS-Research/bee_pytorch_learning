from datetime import datetime
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.autograd import Variable
from torch.optim import Adam
from torchvision.models import alexnet, resnet50
from helper_functions import log_message, annotate_img
import os

from custom_image_dataset import CustomImageDataset
from data_labelling_window import DataLabellingWindow

from supervisor_query_strategies.supervisor_query_strategy import SupervisorQueryStrategy
from supervisor_query_strategies.uncertainty_sampling import UncertaintySampling
from supervisor_query_strategies.margin_of_confidence import MarginOfConfidence

from labelling_query_strategies.labelling_query_strategy import LabellingQueryStrategy
from labelling_query_strategies.CAL import CAL
from labelling_query_strategies.no_labelling import NoLabeling

import torchvision.transforms as T

neural_net_options = [
    "alexnet",
    "resnet50"
]
supervisor_query_options = [
    "uncertainty_sampling",
    "margin_of_confidence"
]
labelling_query_options = [
    "CAL",
    "no_labelling"
]


class ActiveLearner:

    def __init__(self, unlabeled_location: str, output_location: str, neural_net_selection: str,
                 supervisor_query_selection: str, labelling_query_selection: str, training_perc: int,
                 validation_perc: int, cross_validation: bool, classes: List[str], labeled_images_location: str,
                 query_size: int, num_epochs: int, batch_size: Optional[int] = 64):
        self._num_epochs = num_epochs
        self._query_size = query_size
        self._batch_size = batch_size
        self._img_size = 224

        self._use_cross_validation = cross_validation
        self._class_list = classes
        self._annotation_filename = 'annotations.csv'

        self._neural_net_selection = neural_net_selection
        self._supervisor_query_selection = supervisor_query_selection
        self._labelling_query_selection = labelling_query_selection

        self._loss_fn = nn.CrossEntropyLoss()
        self._neural_network: nn = self._build_nn(neural_net_selection)
        self._optimizer = Adam(self._neural_network.parameters(), lr=0.0001, weight_decay=0.0001)
        self._supervisor_query_strategy: SupervisorQueryStrategy = self._build_sqs(supervisor_query_selection)
        self._labelling_query_strategy = self._build_lqs(labelling_query_selection)

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
                    f' outputting model in {os.path.basename(output_location)}.\n\tNeural Network Selection: '
                    f'{neural_net_selection}\n\tSupervisor Query Selection: {supervisor_query_selection}\n\t'
                    f'Labelling Query Selection: {labelling_query_selection}\n\tTraining testing split: {training_perc}'
                    f'/{self._testing_perc}\n\tValidation training split: {validation_perc}/{100 - validation_perc}\n\t'
                    f'Using cross validation: {cross_validation}\n\tAll classes: {classes}',
                    'INFO')
        log_message('Generating training and testing data...', 'DEBUG')
        self._test_set, self._training_set, self._validation_set = self._generate_test_train_validation()

        # self._features = None

        self._figure = None
        self._ax = None

    def save_model(self, model_name=f'model_{datetime.now()}'):
        # Function to save the model
        if not os.path.exists(self._output_location):
            os.makedirs(self._output_location)
        full_path = os.path.join(self._output_location, model_name)
        torch.save(self._neural_network.state_dict(), full_path)

    def _test_accuracy(self, dataset, return_full_results: Optional[bool] = False):
        # Function to test the model with the test dataset and print the accuracy for the test images
        model = self._neural_network
        model.eval()
        accuracy = 0.0
        total = 0.0
        results = []
        features = None
        f_labels = None

        with torch.no_grad():
            for data in dataset:
                images, labels, paths = data
                images = images.float()
                # run the model on the test set to predict labels
                outputs = model(images)
                current_outputs = outputs.cpu().numpy()
                #features = np.concatenate((outputs, current_outputs))
                features = outputs if features is None else np.concatenate((features, outputs))
                # features = np.concatenate((features, cur_features)) if features is not None else cur_features
                f_labels = np.concatenate((f_labels, labels)) if f_labels is not None else labels

                soft = nn.functional.softmax(outputs, dim=1)
                results += (list(zip(list(soft.numpy()), labels.numpy())))
                # log_message(f'{soft.numpy()}', 'WARNING')
                # the label with the highest energy will be our prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                accuracy += (predicted == labels).sum().item()

        # compute the accuracy over all test images
        accuracy = (100 * accuracy / total)
        ret = (accuracy, results, (features, f_labels)) if return_full_results else accuracy

        return ret

    def train(self):
        """
        Training function. We simply have to loop over our data iterator and feed the inputs to the network and
        optimize.
        """
        log_message(f'Starting training over dataset in {self._num_epochs} epochs!')
        model = self._neural_network
        best_accuracy = 0.0

        # Define your execution device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log_message(f"The model will be running on {device} device", 'DEBUG')
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
                                'DEBUG')
                    # zero the loss
                    running_loss = 0.0
            # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
            accuracy = self._test_accuracy(self._validation_set)
            log_message(f'For epoch {epoch + 1} the accuracy over the whole validation set is {accuracy: .3f}%', 'INFO')

            # we want to save the model if the accuracy is the best
            if accuracy > best_accuracy:
                self.save_model()
                best_accuracy = accuracy

        log_message('Finished training!')
        accuracy = self._test_accuracy(self._test_set)
        log_message(f'Accuracy over the whole testing set is {accuracy: .3f}%', 'INFO')
        log_message(f'Model with best validation accuracy has been saved in {os.path.basename(self._output_location)}.')

    def supervisor_query(self):
        """
        Queries the user or 'supervisor' for labels on a number of images equal to `self._query_size`. After getting
        every label, it places the images in the corresponding directories and regenerates the training data.
        """
        # Check to be sure there are some images to label...
        if not os.path.exists(self._unlabeled_images_location) or \
                len(os.listdir(self._unlabeled_images_location)) == 0:
            log_message("No unlabeled images to label!", 'INFO')
        # Load the unlabeled images (needs to run each time this is called, as this method removes unlabeled images.)
        self._unlabeled_images = self._load_unlabeled_images()
        self._unlabeled_set = torch.utils.data.DataLoader(
            self._unlabeled_images, batch_size=self._batch_size, shuffle=True
        )
        # Use the desired supervisor query strategy to find the `n` most useful images to label...
        images = self._supervisor_query_strategy.query_data(self._neural_network, self._unlabeled_set,
                                                            self._query_size)
        # Create the pop-up gui for labelling the images, moving them into the correct location.
        labelling_window = DataLabellingWindow(self._class_list, images, self._labeled_images_location,
                                               self._unlabeled_images_location)
        labelling_window.run_gui()
        # Reload labeled images dataset with newly labeled images
        self._labeled_images = self._load_labeled_images()
        # Reload unlabeled images now that some have been removed...
        self._unlabeled_images = self._load_unlabeled_images()
        self._unlabeled_set = torch.utils.data.DataLoader(
            self._unlabeled_images, batch_size=self._batch_size, shuffle=True
        )

    def _labelling_query(self):
        pass

    def _generate_test_train_validation(self):
        if self._labeled_images is None:
            log_message(f'No labeled images found. Querying {self._query_size} random images for you to label...')
            self.supervisor_query()
        total_count = int(len(self._labeled_images))
        train_count = int(total_count * (self._training_perc / 100))
        test_count = total_count - train_count
        valid_count = int(train_count * (self._validation_perc / 100))
        train_count -= valid_count
        training_set, validation_set, test_set = torch.utils.data.random_split(
            self._labeled_images, (train_count, valid_count, test_count)
        )

        log_message('Split labeled data into testing, training and validation...', 'DEBUG')
        log_message(f'{"Not u" if self._use_cross_validation is False else "U"}sing cross validation to generate the '
                    f'validation set.', 'DEBUG')
        if self._use_cross_validation:
            log_message('Cross validation has not been implemented yet.', 'WARNING')

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
                    log_message('Finding labeled images...', 'DEBUG')
                path = os.path.join(self._labeled_images_location, nn_class)
                if os.path.exists(path):
                    log_message(f'Found labeled images for class "{nn_class}"!', 'DEBUG')
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
            log_message(f'Finding unlabeled images...', 'DEBUG')
            annotations_fp = os.path.join(self._unlabeled_images_location, self._annotation_filename)
            if os.path.exists(annotations_fp):
                os.remove(annotations_fp)
            for filepath in os.listdir(self._unlabeled_images_location):
                annotate_img(annotations_fp, os.path.basename(filepath), -1)
            unlabeled_images = CustomImageDataset(annotations_fp, self._unlabeled_images_location,
                                                  transform=T.Resize((self._img_size, self._img_size)))

        return unlabeled_images

    def _build_nn(self, neural_net_selection: str) -> nn:
        neuralnet: nn
        if neural_net_selection == 'alexnet':
            neuralnet = alexnet(pretrained=True)
            layer = neuralnet.classifier
        elif neural_net_selection == 'resnet50':
            neuralnet = resnet50(pretrained=True)
            layer = neuralnet.fc
        else:
            log_message(f'Sorry, the given neural network, \"{neural_net_selection}\", has not been implemented.',
                        'ERROR')
            raise ValueError()
        if type(layer) == nn.modules.container.Sequential:
            neuralnet.classifier[-1] = nn.Linear(in_features=layer[-1].in_features, out_features=len(self._class_list))
        else:
            neuralnet.classifier = nn.Linear(in_features=layer.in_features, out_features=len(self._class_list))
        return neuralnet

    @staticmethod
    def _build_sqs(supervisor_query_selection: str) -> SupervisorQueryStrategy:
        supervisor_query_strategy: SupervisorQueryStrategy
        if supervisor_query_selection == "uncertainty_sampling":
            supervisor_query_strategy = UncertaintySampling()
        elif supervisor_query_selection == "margin_of_confidence":
            supervisor_query_strategy = MarginOfConfidence()
        else:
            log_message(f'Sorry, the given supervisor query strategy, \"{supervisor_query_selection}\", '
                        f'has not been implemented.',
                        'ERROR')
            raise ValueError()
        return supervisor_query_strategy

    @staticmethod
    def _build_lqs(labelling_query_selection: str) -> LabellingQueryStrategy:
        labelling_query_strategy: LabellingQueryStrategy
        if labelling_query_selection == "CAL":
            labelling_query_strategy = CAL()
        elif labelling_query_selection == "no_labelling":
            labelling_query_strategy = NoLabeling()
        else:
            log_message(f'Sorry, the given labelling query strategy, \"{labelling_query_selection}\", '
                        f'has not been implemented.',
                        'ERROR')
            raise ValueError()
        return labelling_query_strategy

    @staticmethod
    def _scale_to_01_range(x):
        value_range = (np.max(x) - np.min(x))
        starts_from_zero = x - np.min(x)
        return starts_from_zero / value_range

    def get_figure(self) -> plt.Figure:
        if self._figure is None:
            figure, ax = plt.subplots(figsize=(10, 10), dpi=100)
            self._figure = figure
            self._ax = ax
        self._ax.cla()
        self._ax.set_xlabel(f'Probability {self._class_list[1]}')
        self._ax.set_ylabel(f'Probability {self._class_list[0]}')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        accuracy, results, _ = self._test_accuracy(self._test_set, True)
        points, labels = zip(*results)
        x, y = zip(*points)
        self._ax.grid()
        for class_ind in np.unique(labels):
            i = np.where(labels == class_ind)
            self._ax.scatter(np.asarray(x)[i], np.asarray(y)[i], label=self._class_list[class_ind], alpha=0.7)

        accuracy, results = self._test_accuracy(self._unlabeled_set, True)
        points, labels = zip(*results)
        x, y = zip(*points)
        self._ax.scatter(x, y, label='Unlabeled', alpha=0.4)
        self._ax.legend()
        self._figure.canvas.draw()
        # plt.legend()
        return self._figure

    def get_tsne(self) -> plt.Figure:
        if self._figure is None:
            figure, ax = plt.subplots(figsize=(10, 10), dpi=100)
            self._figure = figure
            self._ax = ax
        self._ax.cla()
        t_accuracy, t_results, (t_features, tf_labels) = self._test_accuracy(self._test_set, True)
        u_accuracy, u_results, (u_features, uf_labels) = self._test_accuracy(self._unlabeled_set, True)
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
