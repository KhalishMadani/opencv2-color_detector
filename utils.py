import numpy as np
import cv2

def get_limits(color):

    # convert the input color into 8-bit unsigned integer array which is required by OpenCv functions
    # then, convert the BGR(Blue, Green, Red) color to HSV (Hue, Saturation, Value) color space
    c = np.uint8([[color]]) 
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    # extract the hue value from the converted HSV color
    hue = hsvC[0][0][0]
    print(f'hue: {hue}')

    # Determine the color range by setting the lower and upper limits for the hue value
    # based on the extracted hue. This helps in defining a range for color detection.
    if hue >= 165:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    
    elif hue <= 15: 
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    
    else:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit