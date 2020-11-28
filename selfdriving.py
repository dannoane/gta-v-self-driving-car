import numpy as np
from statistics import mean
from PIL import ImageGrab
import cv2
from util import timers
from gtav import gtav_input

def gamma_correction(image, gamma = 1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)

def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [ vertices ], 255)
    masked = cv2.bitwise_and(image, mask)

    return masked

def draw_lines(image, lines, color = [255, 255, 255], thickness = 3):
    if lines is None:
        return

    for line in lines:
        # because line is an array with just one element (another array)
        coords = line[0]
        try:
            cv2.line(image, (coords[0], coords[1]), (coords[2], coords[3]),
                color=color, thickness=thickness)
        except Exception:
            continue

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
        return None, None

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
        return None, None

    lane1_id = top_lanes[0][0]
    lane2_id = top_lanes[1][0]

    l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
    l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])

    return [l1_x1, l1_y1, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2]

def process_image(image):
    original_img = image
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # convert image to grayscale
    # this way each pixel takes only 8 bits instead of 24
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # make the algorithm more suitable for night driving by
    # improving the contrast in the image
    processed_img = cv2.equalizeHist(processed_img)
    # slightly reduce the contrast so that the image is not too bright
    processed_img = gamma_correction(processed_img, 0.7)
    
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
        minLineLength=100, maxLineGap=5)
    draw_lines(processed_img, lines)

    l1, l2 = find_lanes(lines)
    draw_lines(original_img, [[l1], [l2]], [0, 255, 0], 30)

    return original_img, processed_img

timer = timers.Timers()
gameinput = gtav_input.GtaVInput()
while (True):
    timer.start('per frame')
    
    # grab screen
    printscr = ImageGrab.grab(bbox=(0, 40, 800, 640))
    # transform to 3d array
    printscr_np = np.array(printscr, dtype='uint8')\
        .reshape((printscr.size[1], printscr.size[0], 3))

    original_image, processed_image = process_image(printscr_np)

    timer.end('per frame')

    # display capture
    # cv2.imshow('window', cv2.cvtColor(printscr_np, cv2.COLOR_BGR2RGB))
    cv2.imshow('window', original_image)

    # wait at least 25ms for a key event
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break