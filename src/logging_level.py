from enum import Enum


class LoggingLevel(Enum):
    """
    Enumerated type of all the logging levels and their respective colors on the GUI
    """
    DEBUG = 'lightsteelblue'
    INFO = 'white'
    WARNING = 'yellow'
    ERROR = 'red'
