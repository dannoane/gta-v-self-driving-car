import numpy as np
from PIL import ImageGrab
import cv2
from util import timers
from image_processing.image import Image, ColorMask, ROIMask
from image_processing.find_lanes import find_lanes
from gtav import gtav_input
import time

def process_image(image: Image):
    original = image.copy().bgr_to_rgb()
    grayscale = image.copy().bgr_to_gray()

    # 1. convert image to grayscale
    # this way each pixel takes only 8 bits instead of 24
    # and the image can be used with the Canny and Hough Lines algorithms
    # 2. isolate yellow and white lane lines
    # 3. make the algorithm more suitable for night driving by
    # improving the contrast in the image using the histogram equalization
    # and slightly reduce the contrast after, using gamma correction so that the
    # image is not too bright
    # 4. blur the image. suppress noise in Canny Edge Detection by averaging out the
    # pixel values in a neighborhood
    # 5. find edges in the image using the Canny algorithm
    # 6. find line segments in the image using the Hough Lines algorithms
    # returns an array of arrays. each nested array contains the (x, y)
    # coordinates of the ends of the segments found

    lower_yellow = np.array([0, 130, 130], dtype='uint8')
    upper_yellow = np.array([180, 255, 255], dtype='uint8')
    # leave the yellow and the white colored items
    color_mask = ColorMask()\
        .add_color(grayscale.get_image(), [200, 255])\
        .add_color(original.get_image(), [lower_yellow, upper_yellow])\
        .build()

    # TODO: find better outline for when driving a car in 1st person
    # or switch to 3rd person
    roi_mask = ROIMask([[10, 500], [10, 300], [300, 200], [500, 200], [800, 300], [800, 500]])\
        .build(grayscale.get_image())

    processed = grayscale\
        .copy()\
        .apply_mask(color_mask)\
        .equalize_hist()\
        .gamma_correction(0.9)\
        .canny(threshold1=200, threshold2=300)\
        .gaussian_blur(kernel_size=(5, 5), sigma_x=0)\
        .apply_mask(roi_mask)

    lines = processed.hough_lines_p(rho=1, theta=np.pi/180, threshold=180, 
        min_line_length=100, max_line_gap=5)

    l1, l2, m1, m2 = find_lanes(lines)
    original.draw_lines([[l1], [l2]], [0, 255, 0], 20)

    return original, processed, m1, m2

training_data = []
time.sleep(5)

timer = timers.Timers()
gameinput = gtav_input.GtaVInput()
while (True):
    timer.start('per frame')
    
    # grab screen
    printscr = ImageGrab.grab(bbox=(0, 40, 800, 640))
    image = Image.from_screen_caputre(printscr)
    
    original_image, processed_image, m1, m2 = process_image(image)

    if abs(m1) < 0.15:
        m1 = 0
    if abs(m2) < 0.15:
        m2 = 0
    if m1 < 0 and m2 < 0:
        gameinput.steer_right(release=True)
    elif m1 > 0 and m2 > 0:
        gameinput.steer_left(release=True)
    elif m1 == 0 and m2 == 0:
        gameinput.move_forward(release=True)
        # gameinput.hand_break(release=True)
    else:
        gameinput.move_forward(release=True)

    # display capture
    cv2.imshow('window', original_image.get_image())
    # cv2.imshow('window2', processed_image.get_image())

    timer.end('per frame')

    # wait at least 25ms for a key event
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break