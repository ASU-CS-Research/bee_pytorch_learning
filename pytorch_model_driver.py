from label_data import LabelData
from image_neural_network import Network
from model_utils import ModelUtils
import torch
from torchvision.models import alexnet

import sys
import os
import random


def get_best_images(num_samples: int, unl_filepaths, model_utils=None):
    if model_utils is None:
        random.shuffle(unl_filepaths)
    else:
        unl_filepaths = model_utils.return_least_confident()

    if num_samples <= len(unl_filepaths):
        unl_filepaths = unl_filepaths[0:num_samples]
    # print(len(unl_filepaths))
    return unl_filepaths


if __name__ == '__main__':
    num_samples = (sys.argv[1] if len(sys.argv) > 1 else 50)

    unlabeled_image_location = os.path.abspath('./unlabeled')
    unlabeled_img_filepaths = os.listdir(unlabeled_image_location)
    # unlabeled_img_filepaths = get_best_images(num_samples, unlabeled_img_filepaths)

    classes_list = ["abdomen", "head_and_thorax", "bad_image"]
    # network = Network(classes_list)
    network = alexnet(pretrained=False)
    model_utils = ModelUtils(network)
    model_utils.train(num_epochs=10)

    unlabeled_img_filepaths = get_best_images(num_samples, unlabeled_img_filepaths, model_utils)
    # print(unlabeled_img_filepaths)
    label_data = LabelData(classes_list, unlabeled_img_filepaths)

    while True:
        label_data.run_gui()
        model_utils.train(num_epochs=5)
        unlabeled_img_filepaths = get_best_images(num_samples, unlabeled_img_filepaths, model_utils)
        label_data.add_images(unlabeled_img_filepaths)
