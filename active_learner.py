from typing import List

import torch.nn as nn
from sklearn.model_selection import train_test_split
from torchvision.models import alexnet, resnet50
from log_central import log_message, annotate_img
import os

from custom_image_dataset import CustomImageDataset

from supervisor_query_strategies.supervisor_query_strategy import SupervisorQueryStrategy
from supervisor_query_strategies.uncertainty_sampling import UncertaintySampling

from labelling_query_strategies.labelling_query_strategy import LabellingQueryStrategy
from labelling_query_strategies.CAL import CAL
from labelling_query_strategies.no_labelling import NoLabeling

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
                 query_size):
        self._unlabeled_images_location = unlabeled_location
        self._output_location = output_location
        self._neural_net_selection = neural_net_selection
        self._supervisor_query_selection = supervisor_query_selection
        self._labelling_query_selection = labelling_query_selection
        self._neural_network: nn = self._build_nn(neural_net_selection)
        self._training_perc = training_perc
        self._testing_perc = 100 - training_perc
        self._validation_perc = validation_perc
        self._labeled_images_location = labeled_images_location if labeled_images_location is not None \
            else os.path.abspath('./data/labeled')
        self._supervisor_query_strategy: SupervisorQueryStrategy = self._build_sqs(supervisor_query_selection)
        self._labelling_query_strategy = self._build_lqs(labelling_query_selection)
        self._use_cross_validation = cross_validation
        self._class_list = classes
        self._annotation_filename = 'annotations.csv'
        self._query_size = query_size

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

    def train(self):
        pass

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
        training_set, test_set = train_test_split(self._labeled_images, train_size=self._training_perc / 100)
        log_message('Split labeled data into testing and training...', 'DEBUG')
        log_message(f'{"Not u" if self._use_cross_validation is False else "U"}sing cross validation to generate the '
                    f'validation set.', 'DEBUG')
        if self._use_cross_validation:
            log_message('Cross validation has not been implemented yet.', 'WARNING')
        training_set, validation_set = train_test_split(training_set, test_size=self._validation_perc / 100)
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
            labeled_images = CustomImageDataset(annotations_fp, self._labeled_images_location)

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
            unlabeled_images = CustomImageDataset(annotations_fp, self._unlabeled_images_location)

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
