from datetime import datetime
from typing import List

import PySimpleGUI as sg
import os

import active_learner_intermediary
from helper_functions import log_message, draw_figure
from logging_level import LoggingLevel

NUM_EPOCHS = 5
QUERY_SIZE = 50
BATCH_SIZE = 64
DEFAULT_OUTPUT_DIR = os.path.abspath(f'./outputs/output_{datetime.now().strftime("%Y-%m-%d")}')

font = ("Helvetica", 18)
sg.set_options(font=font, text_element_background_color='white', text_color='black')

images_to_label: List[str] = []
classes: List[str] = []

"""""""""
///  LAYOUTS  ///
"""""""""
layouts = [[] for _ in range(10)]

layouts[0] = [
    [sg.T('Labeled Images Directory: '), sg.FolderBrowse('Select File Location...',
                                                         initial_folder=os.path.abspath('../'),
                                                         key='labeled_dir')],
    [sg.Text('Unlabeled Images Directory: '), sg.FolderBrowse('Select File Location...',
                                                              initial_folder=os.path.abspath('../'),
                                                              key='unlabeled_dir')],
    [sg.Text('Output Folder: '), sg.FolderBrowse('Select File Location...',
                                                 initial_folder=os.path.abspath('../'),
                                                 key='output_dir')],
    [sg.Text('Number of epochs between supervisor query: '), sg.InputText(str(NUM_EPOCHS), key='num_epochs')],
    [sg.Text('Classes to train: '), sg.InputText(key='classes')],
    [sg.Text('Classes should be given in the form \'Class0, Class1, ... ClassN\'')],
    [sg.Text('Query Size: '), sg.InputText(str(QUERY_SIZE), key='query_size')],
    [sg.Text('Query Size is the number of images to ask for labels after each training session.')],
    [sg.Button('Train', key='train')]
]

layouts[1] = [
    [sg.Button('Retrain Model', key='retrain'),
     sg.Button('Supervisor Query', key='supervisor_query_button'),
     sg.Button('Labelling Query', key='labelling_query_button'),
     sg.Button('Create Graph', key='create_graph')],
    [sg.Multiline(size=(100, 30),
                  background_color='black',
                  text_color='white',
                  reroute_cprint=True,
                  autoscroll=True,
                  auto_refresh=True,
                  echo_stdout_stderr=True,
                  key='output_box')]
]

layouts[2] = [
    [sg.T('This page is for some advanced decisions for further customization.')],
    [sg.T('Neural Network Selection: '), sg.Listbox(active_learner_intermediary.neural_net_options,
                                                    default_values=active_learner_intermediary.neural_net_options[0],
                                                    key='neural_network')],
    [sg.T('Supervisor Query Selection: '), sg.Listbox(active_learner_intermediary.supervisor_query_options,
                                                      default_values=
                                                      active_learner_intermediary.supervisor_query_options[0],
                                                      key='supervisor_query')],
    [sg.T('Batch Size: '), sg.InputText(str(BATCH_SIZE), key='batch_size')],
    [sg.T('Training percentage: '), sg.InputText(70, key='training_perc'), sg.T('%')],
    [sg.T('Remaining percent of the data used for testing.')],
    [sg.T('Validation Percentage: '), sg.InputText(20, key='validation_perc'), sg.T('%')],
    [sg.T('Validation set is taken as a percentage of the training set.')],
    [sg.T('Cross Validation'), sg.Checkbox('Enabled', default=True, key='cross_validation')]
]

layouts[3] = [
    [sg.T('T-SNE Representation')],
    [sg.Canvas(key='plt_canvas')]
]

tab_names = ['Home', 'Output', 'Advanced', 'T-SNE']


tabs = [sg.Tab(tab_name, layouts[i].copy(), title_color='Black', border_width=10, background_color='White',
               element_justification='Left') for i, tab_name in enumerate(tab_names)]
tabgrp = [
    [sg.TabGroup([tabs], tab_location='centertop',
                 title_color='Black', tab_background_color='Gray', selected_title_color='Black',
                 selected_background_color='White', border_width=5, key='tab_selection')]]

"""
///  END LAYOUTS ///
"""

# Define Window
window = sg.Window("Active Learning GUI", tabgrp, size=(1024, 768))


def run_gui():

    activelearner = None
    while True:
        # Read  values entered by user
        event, values = window.read(timeout=100)

        if not event == sg.TIMEOUT_EVENT and event is not None:
            log_message(f'event: {event}', LoggingLevel.DEBUG, window)
            draw_figure(canvas=window['plt_canvas'].TKCanvas)

        if event == 'train' or event == 'retrain':
            window['Output'].select()
            if activelearner is None:
                try:
                    values['num_epochs'] = int(values['num_epochs'])
                    values['batch_size'] = int(values['batch_size'])
                    values['validation_perc'] = int(values['validation_perc'])
                    values['training_perc'] = int(values['training_perc'])
                    values['query_size'] = int(values['query_size'])
                except ValueError as e:
                    log_message(f'The number of epochs, training percent and validation percent, the query size, and '
                                f'the batch size must all be integers. {e}', LoggingLevel.ERROR)
                    continue
                if values['unlabeled_dir'] == '':
                    if values['labeled_dir'] == '':
                        log_message(f'No directory location provided for labeled or unlabeled images. With no training '
                                    f'data, we can\'t train, so this training session is aborted.', LoggingLevel.ERROR)
                        continue
                if values['output_dir'] == '':
                    log_message(f'No directory location provided for output directory. Defaulting to {DEFAULT_OUTPUT_DIR}',
                                LoggingLevel.WARNING)
                    values['output_dir'] = DEFAULT_OUTPUT_DIR
                values['classes'] = values['classes'].split(', ')

                activelearner = active_learner_intermediary.build_active_learner(values['unlabeled_dir'],
                                                                                 values['output_dir'],
                                                                                 values['neural_network'][0],
                                                                                 values['supervisor_query'][0],
                                                                                 values['training_perc'],
                                                                                 values['validation_perc'],
                                                                                 values['cross_validation'],
                                                                                 values['classes'],
                                                                                 values['labeled_dir'],
                                                                                 values['query_size'],
                                                                                 values['num_epochs'],
                                                                                 values['batch_size'])
            activelearner.train()
        if event == 'supervisor_query_button':
            if activelearner is not None:
                try:
                    values['query_size'] = int(values['query_size'])
                except ValueError as e:
                    log_message(f'Value for query size is not an int. Aborting supervisor query. Error output: {e}',
                                LoggingLevel.ERROR)
                    continue
                activelearner.update_query_size(values['query_size'])
                activelearner.supervisor_query()
            else:
                # For right now... but later implement something here that allows the user to get some random images and
                # then train a model on those and continue execution.
                log_message('A model needs to be built for the active learner to find images in the region of '
                            'disagreement, so aborting supervisor query.', LoggingLevel.ERROR)
        if event == 'labelling_query_button':
            if activelearner is not None:
                activelearner.labelling_query()
            else:
                # For right now... but later implement something here that allows the user to get some random images and
                # then train a model on those and continue execution.
                log_message('A model needs to be built for the active learner to find images in the region of '
                            'disagreement, so aborting labelling query.', LoggingLevel.ERROR)
        if event == 'create_graph':
            if activelearner is not None:
                # fig = activelearner.get_figure()
                fig = activelearner.get_tsne()
                draw_figure(figure=fig)
        if event == sg.WIN_CLOSED or event == 'Close':
            break

    window.close()
