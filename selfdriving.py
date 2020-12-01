import numpy as np
from statistics import mean
from PIL import ImageGrab
import cv2
from util import timers
from image_processing.image import Image, ColorMask, ROIMask
from gtav import gtav_input
import time

def average_lane(lane_data):
    x1s = []
    y1s = []
    x2s = []
    y2s = []
    
    for data in lane_data:
        lane = data[2]
        x1s.append(lane[0])
        y1s.append(lane[1])
        x2s.append(lane[2])
        y2s.append(lane[3])
    
    return int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s)) 

def find_lanes(lines):
    if lines is None:
        return None, None, 0, 0

    # find the maxium y value for a lane marker
    # since we cannot assume the horizon will always be at the same point
    min_y = 10**100
    max_y = 600 # height of the screen
    for line in lines:
        for coords in line:
            current_min = min(coords[1], coords[3])
            if current_min < min_y:
                min_y = current_min

    new_lines = []
    line_dict = dict()

    for index, line in enumerate(lines):
        if line is None or len(line) < 1:
            continue

        for coords in line:
            if coords is None or len(coords) < 4:
                continue

            x_coords = (coords[0], coords[2])
            y_coords = (coords[1], coords[3])
            A = np.vstack([x_coords, np.ones(len(x_coords))]).T
            m, b = np.linalg.lstsq(A, y_coords)[0]

            x1 = (min_y - b) / m
            x2 = (max_y - b) / m

            if x1 == np.Infinity or x2 == np.Infinity:
                continue

            new_line = [int(x1), min_y, int(x2), max_y]
            line_dict[index] = [m, b, new_line]
            new_lines.append(new_line)

    final_lanes = dict()
    for index in line_dict:
        final_lanes_copy = final_lanes.copy()
        m = line_dict[index][0]
        b = line_dict[index][1]
        line = line_dict[index][2]

        if len(final_lanes) == 0:
            final_lanes[m] = [[m, b, line]]
        else:
            found_copy = False

            for other_ms in final_lanes_copy:
                if not found_copy:
                    if abs(other_ms * 1.2) > abs(m) > abs(other_ms * 0.8):
                        if abs(final_lanes_copy[other_ms][0][1] * 1.2) > abs(b) > abs(final_lanes_copy[other_ms][0][1] * 0.8):
                            final_lanes[other_ms].append([m, b, line])
                            found_copy = True
                            break
                    else:
                        final_lanes[m] = [[m, b, line]]
            
    line_counter = {}
    for lanes in final_lanes:
        line_counter[lanes] = len(final_lanes[lanes])

    top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1][:2]

    if len(top_lanes) < 2:
        return None, None, 0, 0

    lane1_id = top_lanes[0][0]
    lane2_id = top_lanes[1][0]

    l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
    l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])

    return [l1_x1, l1_y1, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2], lane1_id, lane2_id

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

timer = timers.Timers()
gameinput = gtav_input.GtaVInput()
time.sleep(5)
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

    timer.end('per frame')

    # display capture
    cv2.imshow('window', original_image.get_image())
    # cv2.imshow('window2', processed_image.get_image())

    # wait at least 25ms for a key event
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break