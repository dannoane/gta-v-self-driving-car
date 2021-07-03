from statistics import mean
import numpy as np

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

    # find the minimum y value for a lane marker
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