from typing import List, Optional

import PySimpleGUI as sg
from helper_functions import log_message
import os
from torch import nn
import torchvision.transforms as T
from torchvision.io import read_image
from torch.autograd import Variable
from torch import no_grad


class ExamineImagesPopup:
    def __init__(self, class_list: List[str], img_filenames: List[str], output_location: str,
                 unlabeled_imgs_location: str, model: nn, transform: T):
        self._img_index = 0
        self._model_output_str = 'Model output for image: '
        self._layout = [
            [sg.T('Image number: '), sg.InputText(str(self._img_index), key='img_index')],
            [sg.T(f'Out of {len(img_filenames)} images.')],
            [sg.B('←'), sg.B('→')],
            [sg.T(self._model_output_str, key='model_output')],
            [sg.Image(key='current_image', size=(100, 100))],
            [sg.T('Look through the images and confirm that it is working as intended, and then accept or reject'
                  'the labelling query.')],
            [sg.B('Accept'), sg.T('Probability Threshold: '), sg.InputText(str(0), key='probability_threshold'),
             sg.T('%')],
            [sg.B('Reject')]
        ]
        self._class_list = class_list
        self._img_filenames = img_filenames
        self._output_location = os.path.abspath(output_location)
        self._unlabeled_imgs_location = os.path.abspath(unlabeled_imgs_location)
        self._img_list = [(os.path.join(self._unlabeled_imgs_location, file_name),
                          read_image(os.path.join(self._unlabeled_imgs_location, file_name)))
                          for file_name in img_filenames]
        self._model = model
        self._transform = transform
        self._outputs = [None for _ in img_filenames]

    def _get_model_output(self, img, img_idx: Optional[int] = None):
        if img_idx is None:
            img_idx = self._img_index
        if self._outputs[img_idx] is None:
            img = Variable(img.float().cpu())
            img = self._transform(img).unsqueeze(0)
            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self._model.to("cpu")
            model_output = self._model(img)
            model_output = nn.functional.softmax(model_output, dim=1)
            self._outputs[img_idx] = model_output.numpy()
        else:
            model_output = self._outputs[img_idx]
        return model_output

    def _accept_query(self, thresh):
        n_labeled_images = 0
        to_label = [[] for _ in self._class_list]
        for i, (path, img) in enumerate(self._img_list):
            probs = self._get_model_output(img, i)[0]
            if max(probs) > thresh:
                to_label[list(probs).index(max(probs))].append(path)
        for i, fp_by_class in enumerate(to_label):
            class_location = os.path.join(self._output_location, f'{self._class_list[i]}')
            if not os.path.exists(class_location):
                os.makedirs(class_location)
            for path in fp_by_class:
                n_labeled_images += 1
                os.rename(path,
                          os.path.join(class_location, os.path.basename(path)))
        return n_labeled_images

    def run_gui(self):
        window = sg.Window('Data queried!', self._layout, finalize=True, modal=True)
        values = None
        with no_grad():
            while True:
                if values is not None and values['img_index'].isnumeric() and \
                        int(values['img_index']) < len(self._img_list):
                    self._img_index = int(values['img_index'])
                filepath, img = self._img_list[self._img_index]
                window['current_image'].update(filepath)
                model_output = self._get_model_output(img)
                output_str = self._model_output_str
                for i, class_name in enumerate(self._class_list):
                    output_str += f'{class_name}: {model_output[0][i] * 100: .2f}%, '
                window['model_output'].update(output_str)

                event, values = window.read(timeout=100)
                if event == '→':
                    self._img_index += 1
                    window['img_index'].update(self._img_index)
                elif event == '←':
                    self._img_index -= 1
                    window['img_index'].update(self._img_index)
                elif event == 'Accept':
                    try:
                        n_labeled_images = self._accept_query(thresh=float(values['probability_threshold']) / 100)
                        log_message(f'Finished labelling query! labeled {n_labeled_images} images.', 'INFO')
                    except ValueError as e:
                        log_message('Probability threshold must be able to be converted into a float. '
                                    f'Error output: {e}', 'ERROR')
                    break

                elif event == sg.WIN_CLOSED or event == 'Close' or event == 'Reject':
                    log_message('Labeling query rejected.', 'INFO')
                    break

