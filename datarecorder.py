import numpy as np
from PIL import ImageGrab
import cv2
from util import timers, capture_keys
from image_processing.image import Image
from gtav import gtav_input
import os

def load_training_data():
    file_name = 'training_data.npy'

    training_data = []
    if os.path.isfile(file_name):
        print('[+] Found previous data. Loading...')
        training_data = list(np.load(file_name, allow_pickle=True))
    else:
        print('[-] No previous training data found.')

    return training_data

def dump_training_data(training_data):
    file_name = 'training_data.npy'
    np.save(file_name, training_data)

def keys_to_output(keys):
    # check only A, W and D
    output = [0, 0, 0]
    
    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1

    return output

training_data = []
training_data = load_training_data()

timer = timers.Timers()
gameinput = gtav_input.GtaVInput()
while (True):
    timer.start('per frame')
    
    # grab screen
    printscr = ImageGrab.grab(bbox=(0, 40, 800, 640))
    image = Image.from_screen_caputre(printscr)
    
    processed = image\
        .copy()\
        .bgr_to_gray()\
        .resize((320, 240))

    keys = capture_keys.key_check()
    output = keys_to_output(keys)

    training_data.append([processed.get_image(), output])

    if len(training_data) % 500 == 0:
        print('....')
        dump_training_data(training_data)

    cv2.imshow('window', processed.get_image())

    timer.end('per frame')

    # wait at least 25ms for a key event
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
