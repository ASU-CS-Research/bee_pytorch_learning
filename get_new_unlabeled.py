import csv

import cv2
from image_tools import ImageTools
import sys
import os


def write_csv(param):
    with open('./unlabeled/annotations.csv', 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([param, 0])


if __name__ == '__main__':
    location = './unlabeled'
    print(f'Running through video at {sys.argv[1]}, finding all bees and putting their parts in {location}.')
    if not os.path.exists(location):
        os.makedirs(location)
    cap = cv2.VideoCapture(sys.argv[1])
    _, frame = cap.read()
    bee_idx = (sys.argv[2] if len(sys.argv) > 3 else 0)
    show = False
    while frame is not None:
        bees, bee_idx = ImageTools.find_bees(frame, show_img=show, bee_index_start=bee_idx)
        for bee in bees:
            cv2.imwrite(os.path.join(location, f'{os.path.basename(sys.argv[1])}bee{bee[2]}part0.png'), bee[0])
            cv2.imwrite(os.path.join(location, f'{os.path.basename(sys.argv[1])}bee{bee[2]}part1.png'), bee[1])
            write_csv(f'{os.path.basename(sys.argv[1])}bee{bee[2]}part0.png')
            write_csv(f'{os.path.basename(sys.argv[1])}bee{bee[2]}part1.png')
        _, frame = cap.read()
        if show:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
