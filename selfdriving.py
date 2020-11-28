import numpy as np
from PIL import ImageGrab
import cv2
from util import timers
from gtav import gtav_input

def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [ vertices ], 255)
    masked = cv2.bitwise_and(image, mask)

    return masked

def draw_lines(image, lines):
    if lines is None:
        return

    for line in lines:
        # because line is an array with just one element (another array)
        coords = line[0]
        cv2.line(image, (coords[0], coords[1]), (coords[2], coords[3]),
            color=[255, 255, 255], thickness=3)

def process_image(image):
    # convert image to grayscale
    # this way each pixel takes only 8 bits instead of 24
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # find edges in the image using the Canny algorithm
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)

    # blur the image
    processed_img = cv2.GaussianBlur(processed_img, ksize=(5, 5), sigmaX=0)

    # TODO: find better outline for when driving a car in 1st person
    # or switch to 3rd person
    road_outline = np.array([[10, 500], [10, 300], [300, 200], [500, 200], [800, 300], [800, 500]])
    processed_img = region_of_interest(processed_img, road_outline)

    # find line segments in the image
    # returns an array of arrays. each nested array contains the (x, y)
    # coordinates of the ends of the segments found
    lines = cv2.HoughLinesP(processed_img, 1, np.pi / 180, 180, 
        minLineLength=30, maxLineGap=15)
    draw_lines(processed_img, lines)

    return processed_img

timer = timers.Timers()
gameinput = gtav_input.GtaVInput()
while (True):
    timer.start('per frame')
    
    # grab screen
    printscr = ImageGrab.grab(bbox=(0, 40, 800, 640))
    # transform to 3d array
    printscr_np = np.array(printscr, dtype='uint8')\
        .reshape((printscr.size[1], printscr.size[0], 3))

    processed_image = process_image(printscr_np)

    timer.end('per frame')

    # display capture
    # cv2.imshow('window', cv2.cvtColor(printscr_np, cv2.COLOR_BGR2RGB))
    cv2.imshow('window', processed_image)

    # wait at least 25ms for a key event
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break