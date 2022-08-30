import PySimpleGUI as sg
import os
import active_learner
from log_central import log_message

NUM_EPOCHS = 5

font = ("Helvetica", 18)
sg.set_options(font=font, text_element_background_color='white', text_color='black')




"""""""""
///  LAYOUTS  ///
"""""""""
layouts = [[] for _ in range(10)]

layouts[0] = [
    [sg.Text('Unlabeled Images Directory: '), sg.FolderBrowse('Select File Location...',
                                                              initial_folder=os.path.abspath('./'),
                                                              key='unlabeled_dir')],
    [sg.Text('Output Folder: '), sg.FolderBrowse('Select File Location...',
                                                 initial_folder=os.path.abspath('./'),
                                                 key='output_dir')],
    [sg.Text('Number of epochs between supervisor query: '), sg.InputText(str(NUM_EPOCHS), key='num_epochs')],
    [],
    [sg.Button('Train')]
]

layouts[1] = [
    [sg.Text('This is where all the logging output is recorded. If you would like to store this logging information'
             'somewhere, click here: ')],
    [sg.Button('Create Logging file', key='log_file')],
    [sg.Multiline(size=(100, 300),
                  background_color='black',
                  text_color='white',
                  # reroute_stdout=True,
                  # reroute_stderr=True,
                  reroute_cprint=True,
                  autoscroll=True,
                  auto_refresh=True,
                  key='output_box')]
]

layouts[2] = [
    [sg.T('This page is for some advanced decisions for further customization.')],
    [sg.T('Neural Network Selection: '), sg.Listbox(active_learner.neural_net_options,
                                                    default_values=active_learner.neural_net_options[0],
                                                    key='neural_network')],
    [sg.T('Supervisor Query Selection: '), sg.Listbox(active_learner.supervisor_query_options,
                                                      default_values=active_learner.supervisor_query_options[0],
                                                      key='supervisor_query')],
    [sg.T('Labelling Query Selection: '), sg.Listbox(active_learner.chosen_query_options,
                                                     default_values=active_learner.chosen_query_options[0],
                                                     key='chosen_query')],
    [sg.T('Training percentage: '), sg.InputText(70, key='training_perc'), sg.T('%')],
    [sg.T('Remaining percent of the data used for testing.')],
    [sg.T('Validation Percentage: '), sg.InputText(20, key='validation_perc'), sg.T('%')],
    [sg.T('Validation set is taken as a percentage of the training set.')]
]

tab_names = ['Home', 'Output', 'Advanced', 'Testing']

tabs = [sg.Tab(tab_name, layouts[i], title_color='Black', border_width=10, background_color='White',
               element_justification='Left') for i, tab_name in enumerate(tab_names)]

# Define Layout with Tabs
tabgrp = [
    [sg.TabGroup([tabs], tab_location='centertop',
                 title_color='Black', tab_background_color='Gray', selected_title_color='Black',
                 selected_background_color='White', border_width=5, key='tab_selection')]]

# Define Window
window = sg.Window("Tabs", tabgrp, size=(1024, 768))

while True:
    # Read  values entered by user
    event, values = window.read(timeout=100)

    if not event == sg.TIMEOUT_EVENT and event is not None:
        log_message(f'event: {event}', 'DEBUG', window)

    if event == 'Train':
        window['Output'].select()
        try:
            values['num_epochs'] = int(values['num_epochs'])
            values['validation_perc'] = int(values['validation_perc'])
            values['training_perc'] = int(values['training_perc'])
        except ValueError as e:
            log_message(f'The number of epochs, training percent and validation percent must be integers. {e}', 'ERROR')
            continue
        if values['unlabeled_dir'] == '' or values['output_dir'] == '':
            log_message(f'Aborting training. File location of unlabeled images and output directory are required!',
                        'ERROR')
            continue

        activelearner = active_learner.ActiveLearner(values['unlabeled_dir'], values['output_dir'],
                                                     values['neural_network'][0], values['supervisor_query'][0],
                                                     values['chosen_query'][0], values['training_perc'],
                                                     values['validation_perc'])
    if event == sg.WIN_CLOSED or event == 'Close':
        break

window.close()