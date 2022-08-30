import torch.nn as nn
from torchvision.models import alexnet, resnet50
from log_central import log_message
import os

from supervisor_query_strategies.supervisor_query_strategy import SupervisorQueryStrategy
from supervisor_query_strategies.uncertainty_sampling import UncertaintySampling

from labelling_query_strategies.labelling_query_strategy import LabellingQueryStrategy
from labelling_query_strategies.CAL import CAL
from labelling_query_strategies.no_labelling import NoLabeling

neural_net_options = [
    "alexnet",
    "resnet50"
]
supervisor_query_options = [
    "uncertainty_sampling"
]
labelling_query_options = [
    "CAL",
    "no_labelling"
]


class ActiveLearner:

    def __init__(self, unlabeled_location, output_location, neural_net_selection: str, supervisor_query_selection: str,
                 labelling_query_selection: str, training_perc: int, validation_perc: int,
                 labeled_images_location: str = None):
        self._unlabeled_location = unlabeled_location,
        self._output_location = output_location,
        self._neural_net_selection = neural_net_selection,
        self._supervisor_query_selection = supervisor_query_selection
        self._labelling_query_selection = labelling_query_selection
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
                    f'Labelling Query Selection: {labelling_query_selection}\n\tTraining testing split: {training_perc}'
                    f'/{self._testing_perc}\n\tValidation training split: {validation_perc}/{100 - validation_perc}',
                    'INFO')

    def train(self):
        pass

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
