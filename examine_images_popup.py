import PySimpleGUI as sg
import cv2

class ExamineImagesPopup:
    def __init__(self, image_paths_list):
        layout = [[sg.Image(key='current_image', size=(100, 100))]]
