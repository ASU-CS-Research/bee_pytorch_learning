import os
from typing import List, Optional

import PySimpleGUI as sg
import cv2
import csv


class LabelData:
     
    def __init__(self, opt_list: List[str], img_list: List[str]):
        self._data_location = os.path.abspath('./data')
        self._data_annotations = os.path.join(self._data_location, 'annotations.csv')
        self._layout = self._create_layout(buttons=opt_list)
        self._opt_list = opt_list
        self._unlabeled_images = os.path.abspath('./unlabeled')
        self._img_list = [(os.path.join(self._unlabeled_images, file_name),
                           cv2.imread(os.path.join(self._unlabeled_images, file_name)))
                          for file_name in img_list]

    def _label_image(self, filepath: str, label: Optional[str] = None):
        if label is None:
            # `None` label means the user wants to delete the image rather than label it.
            os.remove(filepath)
        else:
            if not os.path.exists(os.path.join(self._data_location, 'new_training_data')):
                os.makedirs(os.path.join(self._data_location, 'new_training_data'))
            file_name = os.path.basename(filepath)
            self._write_to_csv(file_name, label)
            os.rename(filepath, os.path.join(self._data_location, 'new_training_data', file_name))

    def _write_to_csv(self, file_name, label):
        with open(self._data_annotations, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([file_name, label])

    def add_images(self, new_img_list):
        for file_name in new_img_list:
            self._img_list.append((os.path.join(self._unlabeled_images, file_name),
                                   cv2.imread(os.path.join(self._unlabeled_images, file_name))))

    def run_gui(self):
        self._layout = self._create_layout(self._opt_list)
        window = sg.Window('Label the data!', self._layout, finalize=True)
        for (filepath, img) in self._img_list:
            window['--IMAGE--'].update(filepath)
            event, values = window.read()
            if event in self._opt_list:
                self._label_image(filepath, self._opt_list.index(event))
            elif event == 'Delete Image':
                self._label_image(filepath)
            elif event == sg.WIN_CLOSED:
                break
        self._img_list = []
        window.close()

    @staticmethod
    def _create_layout(buttons):
        layout = [[sg.Image(key='--IMAGE--')]]
        layout += [[sg.Button(name)] for name in buttons]
        layout.append([sg.Button("Delete Image")])
        return layout
