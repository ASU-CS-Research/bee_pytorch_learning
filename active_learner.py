import torch.nn as nn
from torchvision.models import alexnet
from log_central import log_message
import os

from supervisor_query_strategy import SupervisorQueryStrategy
from uncertainty_sampling import UncertaintySampling

neural_net_options = [
    "alexnet"
]
supervisor_query_options = [
    "uncertainty_sampling"
]
chosen_query_options = [
    "CAL"
]


class ActiveLearner:

    def __init__(self, unlabeled_location, output_location, neural_net_selection: str, supervisor_query_selection: str,
                 chosen_query_selection: str, training_perc: int, validation_perc: int,
                 labeled_images_location: str = None):
        self._unlabeled_location = unlabeled_location,
        self._output_location = output_location,
        self._neural_net_selection = neural_net_selection,
        self._supervisor_query_selection = supervisor_query_selection
        self._chosen_query_selection = chosen_query_selection
        self._neural_network: nn = self._build_nn(neural_net_selection)
        self._training_perc = training_perc
        self._testing_perc = 100 - training_perc
        self._validation_perc = validation_perc
        self._labeled_images_location = labeled_images_location if labeled_images_location is not None \
            else os.path.abspath('./data/labeled')
        self._supervisor_query_strategy: SupervisorQueryStrategy = self._build_sqs(supervisor_query_selection)

        log_message(f'Built active learner. Unlabeled images are in {os.path.basename(unlabeled_location)}, '
                    f'outputting model in {os.path.basename(output_location)}.\n\tNeural Network Selection: '
                    f'{neural_net_selection}\n\tSupervisor Query Selection: {supervisor_query_selection}\n\t'
                    f'Labelling Query Selection: {chosen_query_selection}\n\tTraining testing split: {training_perc}'
                    f'/{self._testing_perc}\n\tValidation training split: {validation_perc}/{100 - validation_perc}',
                    'INFO')

    def train(self):
        pass

    @staticmethod
    def _build_nn(neural_net_selection) -> nn:
        neuralnet: nn = None
        if neural_net_selection == 'alexnet':
            neuralnet = alexnet(pretrained=False)

        return neuralnet

    @staticmethod
    def _build_sqs(supervisor_query_selection) -> SupervisorQueryStrategy:
        supervisor_query_strategy: SupervisorQueryStrategy = None
        if supervisor_query_selection == "uncertainty_sampling":
            supervisor_query_strategy = UncertaintySampling()

        return supervisor_query_strategy
