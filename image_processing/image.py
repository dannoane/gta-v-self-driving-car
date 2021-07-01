import numpy as np
import cv2

class ColorMask:
    def __init__(self):
        self.__colors = []

    def add_color(self, image, color_range):
        self.__colors.append(cv2.inRange(image, color_range[0], color_range[1]))
        return self

    def build(self):
        assert len(self.__colors) > 0
        mask = self.__colors[0]
        for other in self.__colors[1:]:
            mask = cv2.bitwise_or(mask, other)
        return mask

class ROIMask:
    def __init__(self, vertices):
        self.__vertices = np.array(vertices)

    def build(self, image):
        assert len(self.__vertices) > 0
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [self.__vertices], 255)
        return mask

class Image:
    def __init__(self, image):
        self.__image = image

    def get_image(self):
        return self.__image

    def bgr_to_rgb(self):
        self.__image = cv2.cvtColor(self.__image, cv2.COLOR_BGR2RGB)
        return self

    def bgr_to_hsv(self):
        self.__image = cv2.cvtColor(self.__image, cv2.COLOR_BGR2HSV)
        return self

    def bgr_to_gray(self):
        self.__image = cv2.cvtColor(self.__image, cv2.COLOR_BGR2GRAY)
        return self

    def resize(self, size):
        self.__image = cv2.resize(self.__image, size)
        return self

    def apply_mask(self, mask):
        self.__image = cv2.bitwise_and(self.__image, mask)
        return self

    def equalize_hist(self):
        self.__image = cv2.equalizeHist(self.__image)
        return self

    def gamma_correction(self, gamma=1.0):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
		    for i in np.arange(0, 256)]).astype("uint8")

        self.__image = cv2.LUT(self.__image, table)
        return self

    def gaussian_blur(self, kernel_size=(5, 5), sigma_x=0):
        self.__image = cv2.GaussianBlur(self.__image, ksize=kernel_size, sigmaX=sigma_x)
        return self

    def canny(self, threshold1, threshold2):
        self.__image = cv2.Canny(self.__image, threshold1, threshold2)
        return self

    def draw_lines(self, lines, color=[255, 255, 255], thickness=3):
        if lines is None:
            print('[-] No lines sent')
            return self

        for line in lines:
            # because line is an array with just one element (another array)
            coords = line[0]
            if coords is not None:
                try:
                    cv2.line(self.__image, (coords[0], coords[1]), (coords[2], coords[3]),
                        color=color, thickness=thickness)
                except Exception as e:
                    print('[-] {}'.format(e))
                    continue

        return self

    def hough_lines_p(self, rho, theta, threshold, min_line_length, max_line_gap):
        lines = cv2.HoughLinesP(self.__image, rho, theta, threshold,
            minLineLength=min_line_length, maxLineGap=max_line_gap)
        return lines

    def copy(self):
        return Image(self.__image)

    @classmethod
    def from_screen_caputre(self, capture):
        # transform to 3d array
        capture_np = np.array(capture, dtype='uint8')\
            .reshape((capture.size[1], capture.size[0], 3))
        return Image(capture_np)