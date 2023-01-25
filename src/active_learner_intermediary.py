import os
from typing import Optional, List

import torch
import torch.nn as nn
import torchvision.models
from torchvision.models import alexnet, resnet50, mobilenet_v3_small, mobilenet_v3_large, vgg19

from active_learner import ActiveLearner
from helper_functions import log_message
from logging_level import LoggingLevel

from supervisor_query_strategies.supervisor_query_strategy import SupervisorQueryStrategy
from supervisor_query_strategies.uncertainty_sampling import UncertaintySampling

neural_net_options = [
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "resnet50",
    "alexnet",
    "vgg19"
]
supervisor_query_options = [
    "uncertainty_sampling"
]

class_list = []


def build_active_learner(unlabeled_location: str, output_location: str, neural_net_selection: str,
                         supervisor_query_selection: str, training_perc: int,
                         validation_perc: int, cross_validation: bool, classes: List[str], labeled_images_location: str,
                         query_size: int, num_epochs: int, batch_size: Optional[int] = 64,
                         existing_model_location: Optional[str] = None):
    global class_list
    class_list = classes
    if existing_model_location == '':
        existing_model_location = None
    neural_net = _build_nn(neural_net_selection, existing_model_location)
    supervisor_query_strategy = _build_sqs(supervisor_query_selection)
    return ActiveLearner(unlabeled_location, output_location, training_perc, supervisor_query_strategy, neural_net,
                         neural_net_selection, validation_perc, cross_validation, classes, labeled_images_location,
                         query_size, num_epochs, batch_size)


def _build_nn(neuralnet_selection: str, existing_model_location: Optional[str] = None) -> nn:
    global class_list
    neuralnet: nn = None
    if neuralnet_selection == 'alexnet' or neuralnet_selection == 'mobilenet_v3_small' or \
       neuralnet_selection == 'mobilenet_v3_large' or neuralnet_selection == 'vgg19':
        if neuralnet_selection == 'alexnet':
            neuralnet = alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
        elif neuralnet_selection == 'mobilenet_v3_small':
            neuralnet = mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        elif neuralnet_selection == 'mobilenet_v3_large':
            neuralnet = mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        elif neuralnet_selection == 'vgg19':
            neuralnet = vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
        neuralnet.classifier[-1] = nn.Linear(in_features=neuralnet.classifier[-1].in_features,
                                             out_features=len(class_list))
        neuralnet.requires_grad_(False)
        neuralnet.classifier.requires_grad_(True)
    elif neuralnet_selection == 'resnet50':
        neuralnet = resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        neuralnet.fc = nn.Linear(in_features=neuralnet.fc.in_features, out_features=len(class_list))
        neuralnet.requires_grad_(False)
        neuralnet.layer4.requires_grad_(True)
    else:
        log_message(f'Sorry, the given neural network, \"{neuralnet_selection}\", has not been implemented.' +
                    'To implement a new model selection in the user interface, you both have to include a key for it' +
                    'in active_learner_intermediary.neural_net_options as well as adding the configuration in '
                    '_build_nn() in the same file.',
                    LoggingLevel.ERROR)
        raise ValueError()
    if existing_model_location is not None:
        neuralnet.load_state_dict(torch.load(os.path.abspath(existing_model_location)))
    return neuralnet


def _build_sqs(supervisor_query_selection: str) -> SupervisorQueryStrategy:
    supervisor_query_strategy: SupervisorQueryStrategy
    if supervisor_query_selection == "uncertainty_sampling":
        supervisor_query_strategy = UncertaintySampling()
    else:
        log_message(f'Sorry, the given supervisor query strategy, \"{supervisor_query_selection}\", '
                    f'has not been implemented.',
                    LoggingLevel.ERROR)
        raise ValueError()
    return supervisor_query_strategy
