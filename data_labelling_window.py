import os
from typing import List, Optional

import PySimpleGUI as sg
import cv2
from helper_functions import log_message


class DataLabellingWindow:
     
    def __init__(self, class_list: List[str], img_filenames: List[str],
                 data_location: Optional[str] = os.path.abspath('./data'),
                 unlabeled_imgs_location: Optional[str] = os.path.abspath('./unlabeled')):
        self._data_location = data_location
        self._layout = self._create_layout(buttons=class_list)
        self._opt_list = class_list
        self._unlabeled_imgs_location = unlabeled_imgs_location
        self._img_list = [(os.path.join(self._unlabeled_imgs_location, file_name),
                           cv2.imread(os.path.join(self._unlabeled_imgs_location, file_name)))
                          for file_name in img_filenames]

    def _label_image(self, filepath: str, label: Optional[str] = None):
        if label is None:
            # `None` label means the user wants to delete the image rather than label it.
            os.remove(filepath)
        else:
            file_name = os.path.basename(filepath)
            new_location = os.path.join(self._data_location, label)
            if not os.path.exists(new_location):
                os.makedirs(new_location)
            os.rename(filepath, os.path.join(new_location, file_name))

    def add_images(self, new_img_list):
        for file_name in new_img_list:
            self._img_list.append((os.path.join(self._unlabeled_imgs_location, file_name),
                                   cv2.imread(os.path.join(self._unlabeled_imgs_location, file_name))))

    def run_gui(self):
        self._layout = self._create_layout(self._opt_list)
        window = sg.Window('Label the data!', self._layout, finalize=True, modal=True)
        for (filepath, img) in self._img_list:
            window['current_image'].update(filepath)
            event, values = window.read()
            if event in self._opt_list:
                log_message(f'Labelling {os.path.basename(filepath)} as {event}')
                self._label_image(filepath, event)
            elif event == 'Delete Image':
                log_message(f'Deleting {os.path.basename(filepath)}.')
                self._label_image(filepath)
            elif event == sg.WIN_CLOSED:
                break
        self._img_list = []
        window.close()

    @staticmethod
    def _create_layout(buttons):
        layout = [[sg.Image(key='current_image', size=(100, 100))]]
        layout += [[sg.Button(name)] for name in buttons]
        layout.append([sg.Button("Delete Image")])
        return layout
