from typing import Optional, List

import torch
import torch.nn as nn
import torchvision.models
from torchvision.models import alexnet, resnet50

from active_learner import ActiveLearner
from helper_functions import log_message

from supervisor_query_strategies.supervisor_query_strategy import SupervisorQueryStrategy
from supervisor_query_strategies.uncertainty_sampling import UncertaintySampling

neural_net_options = [
    "alexnet",
    "resnet50"
]
supervisor_query_options = [
    "uncertainty_sampling"
]

class_list = []


def build_active_learner(unlabeled_location: str, output_location: str, neural_net_selection: str,
                         supervisor_query_selection: str, training_perc: int,
                         validation_perc: int, cross_validation: bool, classes: List[str], labeled_images_location: str,
                         query_size: int, num_epochs: int, batch_size: Optional[int] = 64):
    global class_list
    class_list = classes
    neural_net = _build_nn(neural_net_selection)
    supervisor_query_strategy = _build_sqs(supervisor_query_selection)
    return ActiveLearner(unlabeled_location, output_location, training_perc, supervisor_query_strategy, neural_net,
                         validation_perc, cross_validation, classes, labeled_images_location, query_size, num_epochs,
                         batch_size)


def _build_nn(neural_net_selection: str) -> nn:
    global class_list
    neuralnet: nn
    if neural_net_selection == 'alexnet':
        neuralnet = alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
        layer = neuralnet.classifier
        neuralnet.classifier[-1] = nn.Linear(in_features=layer[-1].in_features, out_features=len(class_list))
        neuralnet.features.requires_grad_(False)
    elif neural_net_selection == 'resnet50':
        neuralnet = resnet50(pretrained=True)
        layer = neuralnet.fc
        neuralnet.classifier = nn.Linear(in_features=layer.in_features, out_features=len(class_list))
        neuralnet.features.requires_grad_(False)
    else:
        log_message(f'Sorry, the given neural network, \"{neural_net_selection}\", has not been implemented.' +
                    'To implement a new model selection in the user interface, you both have to include a key for it' +
                    'in active_learner_intermediary.neural_net_options as well as adding the configuration in '
                    '_build_nn() in the same file.',
                    'ERROR')
        raise ValueError()

    return neuralnet


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
