from typing import List

import PySimpleGUI as sg
from helper_functions import log_message
import os
from torch import nn
import torchvision.transforms as T
from torchvision.io import read_image
import torch


class ExamineImagesPopup:
    def __init__(self, class_list: List[str], img_filenames: List[str], output_location: str,
                 unlabeled_imgs_location: str, model: nn, transform: T):
        self._img_index = 0
        self._model_output_str = 'Model output for image: '
        self._layout = [
            [sg.T('Image number: '), sg.InputText(str(self._img_index), key='img_index')],
            [sg.T(f'Out of {len(img_filenames)} images.')],
            [sg.B('-'), sg.B('+')],
            [sg.T(self._model_output_str, key='model_output')],
            [sg.Image(key='current_image', size=(100, 100))]
        ]
        self._class_list = class_list
        self._img_filenames = img_filenames
        self._output_location = os.path.abspath(output_location)
        self._unlabeled_imgs_location = os.path.abspath(unlabeled_imgs_location)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._img_list = [(os.path.join(self._unlabeled_imgs_location, file_name),
                          read_image(os.path.join(self._unlabeled_imgs_location, file_name)).to(device))
                          for file_name in img_filenames]
        self._model = model
        self._transform = transform
        self._outputs = [None for _ in img_filenames]

    def run_gui(self):
        window = sg.Window('Data queried!', self._layout, finalize=True, modal=True)
        values = None
        while True:
            if values is not None and values['img_index'].isnumeric() and \
                    int(values['img_index']) < len(self._img_filenames):
                self._img_index = int(values['img_index'])
            filepath, img = self._img_list[self._img_index]
            window['current_image'].update(filepath)
            if self._outputs[self._img_index] is None:
                img = self._transform(img)
                model_output = self._model(img)
                self._outputs[self._img_index] = model_output
            else:
                model_output = self._outputs[self._img_index]
            window['model_output'].update(self._model_output_str + str(model_output))

            event, values = window.read(timeout=100)
            if event == '+':
                self._img_index += 1
                window['img_index'].update(self._img_index)
            elif event == '-':
                self._img_index -= 1
                window['img_index'].update(self._img_index)
            elif event == sg.WIN_CLOSED:
                break
