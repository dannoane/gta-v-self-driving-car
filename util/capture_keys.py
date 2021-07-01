import win32api as wapi
import time

key_list = ['\b']
key_list.extend([chr(x) for x in range(ord('A'), ord('Z'))])
key_list.extend([chr(x) for x in range(ord('1'), ord('9'))])
key_list.extend([' ', ',', '.', '\'', '$', '/', '\\'])

def key_check():
    keys = []
    for key in key_list:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys