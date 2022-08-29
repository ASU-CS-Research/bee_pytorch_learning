from typing import Optional, Tuple, List
import numpy as np
import imutils
import cv2


class ImageTools:

    @staticmethod
    def alter_contrast_brightness(img: np.ndarray, alpha: float, beta: int) -> np.ndarray:
        """

        Args:
            img: Given image to alter the contrast and brightness of.
            alpha: Contrast factor, value is between 1.0 and 3.0
            beta: Brightness factor, value is between 0 and 100
        Returns:
            List[np.ndarray]:
        """
        output_image = np.zeros(img.shape, img.dtype)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                for c in range(img.shape[2]):
                    output_image[y, x, c] = np.clip(alpha * img[y, x, c] + beta, 0, 255)
        return output_image

    @staticmethod
    def sharpen_image(img: np.ndarray, kernel_size: Optional[Tuple[int, int]] = (5, 5),
                      sigma: Optional[float] = 1.0, amount: Optional[float] = 4.0, threshold: Optional[int] = 0):
        """
        Sharpens an image using an unsharp mask. Descriptions for the parameters taken from wikipedia:
        `https://en.wikipedia.org/wiki/Unsharp_masking#Digital_unsharp_masking`
        Args:
            img:
            kernel_size: Gaussian kernel size. kernel_size.width and kernel_size.height can differ but they both must be
              positive and odd.
            sigma: Gaussian kernel standard deviation.
            amount: A percentage that controls the magnitude of each overshoot (how much darker and how much lighter the
              edge borders become). This can also be thought of as how much contrast is added at the edges. It does not
              affect the width of the edge rims.
            threshold: Controls the minimal brightness change that will be sharpened or how far apart adjacent tonal
              values have to be before the filter does anything. This lack of action is important to prevent smooth
              areas from becoming speckled. The threshold setting can be used to sharpen more pronounced edges, while
              leaving subtler edges untouched. Low values should sharpen more because fewer areas are excluded. Higher
              threshold values exclude areas of lower contrast.
        Returns:

        """
        blurred = cv2.GaussianBlur(img, kernel_size, sigma)
        sharpened = float(amount + 1) * img - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(img - blurred) < threshold
            np.copyto(sharpened, img, where=low_contrast_mask)
        return sharpened

    @staticmethod
    def find_bees(img: np.ndarray, lower_hsv: Optional[Tuple[int, int, int]] = (0, 50, 20),
                  upper_hsv: Optional[Tuple[int, int, int]] = (180, 220, 255), show_img: Optional[bool] = False,
                  bee_index_start: Optional[int] = 0) -> Tuple[List[Tuple[np.ndarray, np.ndarray, int]], int]:

        hsv_mask = ImageTools._subtract_background(img, lower_hsv, upper_hsv, show_img)

        contours, _ = cv2.findContours(hsv_mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found_bees = []
        i = bee_index_start
        for contour in contours:
            if len(contour) > 50 and (1300 < cv2.contourArea(contour) < 1850):
                ellipse = cv2.fitEllipse(contour)
                found_bees.append(ImageTools.create_bee(img, contour, i))
                i += 1
                img = cv2.ellipse(img, ellipse, color=(0, 0, 255), thickness=1)
        if show_img:
            cv2.imshow('frame', img)
        return found_bees, i

    @staticmethod
    def _subtract_background(img, lower_hsv, upper_hsv, show_img: bool):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
        if show_img:
            cv2.imshow('hsv mask', mask)
        return mask

    @staticmethod
    def create_bee(img: np.ndarray, contour: np.ndarray, i: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """

        Args:
            img:
            contour:
            i:
        Returns:

        """
        # Use a bounding rectangle to create the bounding area of the contour
        rx, ry, rw, rh = cv2.boundingRect(contour)
        # Then, fit an ellipse to get better information about the shape and direction of the contour
        ellipse = cv2.fitEllipse(contour)
        _, (ma, Ma), angle = ellipse

        # Lastly get the major axis as a series of points.
        r_major = Ma / 2
        if angle > 90:
            angle -= 90
        else:
            angle += 90

        img = img[ry: ry + rh, rx: rx + rw]
        img = imutils.rotate_bound(img, -1 * angle)
        bee_part_0 = img[0: img.shape[1], 0:int(img.shape[0] / 2)]
        bee_part_1 = img[0: img.shape[1], int(img.shape[0] / 2):img.shape[0]]
        return bee_part_0, bee_part_1, i

    @staticmethod
    def color_equals(color1: Tuple[int, int, int] | List[float], color2: Tuple[int, int, int] | List[float],
                     threshold: Optional[int] = 10) -> bool:
        if len(color1) != len(color2):
            return False
        for i in range(len(color1)):
            if color2[i] not in range(int(color1[i] - threshold), int(color1[i] + threshold)):
                return False
        return True

    @staticmethod
    def _average(lst: List[int]) -> float:
        total = sum(lst)
        return total / len(lst)
