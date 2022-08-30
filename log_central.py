from datetime import datetime

color_codes = {
    'DEBUG': 'lightsteelblue',
    'INFO': 'white',
    'WARNING': 'yellow',
    'ERROR': 'red'
}
logger = None


def log_message(message: str, level: str = 'INFO', window=None):
    global logger

    if window is not None:
        logger = window['output_box'].Widget

    logger.tag_config(level, background='black', foreground=color_codes[level])
    ts = datetime.now()
    log_msg = f'{ts} | '
    log_msg += ' %7s | ' % level
    log_msg += message + '\n'
    logger.insert('end', log_msg, level)
