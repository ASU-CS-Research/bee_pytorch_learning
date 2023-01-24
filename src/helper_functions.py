import os
from datetime import datetime
import csv
from typing import Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib

from logging_level import LoggingLevel

matplotlib.use('TkAgg')

logger = None
static_window = None
static_canvas = None
static_fig_agg = None


def log_message(message: str, level: Optional[LoggingLevel] = LoggingLevel.INFO, window=None):
    global logger
    global static_window

    # Build the message
    ts = datetime.now()
    log_msg = f'{ts} | '
    log_msg += ' %7s | ' % level.name
    log_msg += message + '\n'

    # If a PySimpleGUI window is passed through, then set that window as the static_window to update
    if window is not None:
        logger = window['output_box'].Widget
        static_window = window
    # If static_window doesn't exist by this point in the code, then it is being run without the GUI and we should
    # simply print the logging message.
    if static_window is None:
        print(log_msg)
        return

    logger.tag_config(level, background='black', foreground=level.value)

    logger.insert('end', log_msg, level)
    static_window.refresh()


def annotate_img(csv_filepath, basename, class_val):
    if basename == '.DS_Store':
        return
    if not os.path.exists(csv_filepath):
        open_or_append = 'w'
    else:
        open_or_append = 'a'
    with open(csv_filepath, open_or_append) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([basename, class_val])


def draw_figure(canvas=None, figure=None):
    global static_canvas
    if canvas is not None:
        static_canvas = canvas
        return
    else:
        global static_fig_agg
        if static_fig_agg is not None:
            return
        figure_canvas_agg = FigureCanvasTkAgg(figure, static_canvas)
        plot_widget = figure_canvas_agg.get_tk_widget()
        plot_widget.grid(row=0, column=0)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
        static_fig_agg = figure_canvas_agg
        global static_window
        static_window.refresh()

