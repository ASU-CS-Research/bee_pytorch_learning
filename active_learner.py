from datetime import datetime
from typing import List, Optional

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torchvision.models import alexnet, resnet50
from log_central import log_message, annotate_img
import os

from datasplit import DataSplit
from custom_image_dataset import CustomImageDataset

from supervisor_query_strategies.supervisor_query_strategy import SupervisorQueryStrategy
from supervisor_query_strategies.uncertainty_sampling import UncertaintySampling

from labelling_query_strategies.labelling_query_strategy import LabellingQueryStrategy
from labelling_query_strategies.CAL import CAL
from labelling_query_strategies.no_labelling import NoLabeling

import torchvision.transforms as T

neural_net_options = [
    "resnet50",
    "alexnet"
]
supervisor_query_options = [
    "uncertainty_sampling"
]
labelling_query_options = [
    "CAL",
    "no_labelling"
]


class ActiveLearner:

    def __init__(self, unlabeled_location: str, output_location: str, neural_net_selection: str,
                 supervisor_query_selection: str, labelling_query_selection: str, training_perc: int,
                 validation_perc: int, cross_validation: bool, classes: List[str], labeled_images_location: str,
                 query_size, num_epochs: int, batch_size: Optional[int] = 64):
        self._num_epochs = num_epochs
        self._query_size = query_size
        self._batch_size = batch_size
        self._img_size = 224

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

        self._use_cross_validation = cross_validation
        self._class_list = classes
        self._annotation_filename = 'annotations.csv'

        self._labeled_images = self._load_labeled_images()
        self._unlabeled_images = self._load_unlabeled_images()

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

    def save_model(self, model_name=f'model_{datetime.now()}'):
        # Function to save the model
        if not os.path.exists(self._output_location):
            os.makedirs(self._output_location)
        full_path = os.path.join(self._output_location, model_name)
        torch.save(self._neural_network.state_dict(), full_path)

    def _test_accuracy(self, dataset):
        # Function to test the model with the test dataset and print the accuracy for the test images
        model = self._neural_network
        model.eval()
        accuracy = 0.0
        total = 0.0

        with torch.no_grad():
            for data in dataset:
                images, labels, paths = data
                images = images.float()
                # run the model on the test set to predict labels
                outputs = model(images[:, :3, :, :])
                # print(tuple(torch.max(outputs.data, 1)))
                # the label with the highest energy will be our prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                accuracy += (predicted == labels).sum().item()

        # compute the accuracy over all test images
        accuracy = (100 * accuracy / total)
        return accuracy

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
            log_message(f'For epoch {epoch + 1} the accuracy over the whole validation set is {accuracy}%', 'INFO')

            # we want to save the model if the accuracy is the best
            if accuracy > best_accuracy:
                self.save_model()
                best_accuracy = accuracy

        log_message('Finished training!')
        accuracy = self._test_accuracy(self._test_set)
        log_message(f'Accuracy over the whole testing set is {accuracy}', 'INFO')
        log_message(f'Model with best validation accuracy has been saved in {os.path.basename(self._output_location)}.')

    def _supervisor_query(self):
        """
        Queries the user or 'supervisor' for labels on a number of images equal to `self._query_size`. After getting
        every label, it places the images in the corresponding directories and regenerates the training data.
        """

        self._labeled_images = self._load_labeled_images()
        pass

    def _labelling_query(self):
        pass

    def _generate_test_train_validation(self):
        if self._labeled_images is None:
            log_message(f'No labeled images found. Querying {self._query_size} random images for you to label...')
            self._supervisor_query()
        ds = DataSplit(self._labeled_images, train_test_split=self._training_perc / 100,
                       val_train_split=self._validation_perc / 100)
        training_set, validation_set, test_set = ds.get_split(batch_size=self._batch_size)
        log_message('Split labeled data into testing, training and validation...', 'DEBUG')
        log_message(f'{"Not u" if self._use_cross_validation is False else "U"}sing cross validation to generate the '
                    f'validation set.', 'DEBUG')
        if self._use_cross_validation:
            log_message('Cross validation has not been implemented yet.', 'WARNING')

        return test_set, training_set, validation_set

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

    @staticmethod
    def _build_nn(neural_net_selection: str) -> nn:
        neuralnet: nn
        if neural_net_selection == 'alexnet':
            neuralnet = alexnet(pretrained=False)
        elif neural_net_selection == 'resnet50':
            neuralnet = resnet50(pretrained=False)
        else:
            log_message(f'Sorry, the given neural network, \"{neural_net_selection}\", has not been implemented.',
                        'ERROR')
            raise ValueError()
        return neuralnet

    @staticmethod
    def _build_sqs(supervisor_query_selection: str) -> SupervisorQueryStrategy:
        supervisor_query_strategy: SupervisorQueryStrategy
        if supervisor_query_selection == "uncertainty_sampling":
            supervisor_query_strategy = UncertaintySampling()
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
