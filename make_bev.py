# ----------------------------------------------------------------------------
# Copyright (C) [2023] Byounggun Park
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------

import cv2
import numpy as np

imgsz = 256

# Load the image and resize it to the desired size
im0 = cv2.imread('data/demo/lane1.jpg')  # Replace 'test.jpg' with your actual image file path
im0 = cv2.resize(im0, (imgsz, imgsz))

# Initial source points - define a trapezoid
src = np.float32([[0, imgsz],  # Bottom left
                  [imgsz, imgsz],  # Bottom right
                  [imgsz*0.6, imgsz*0.6],  # Top right
                  [imgsz*0.4, imgsz*0.6]])  # Top left

# Destination points - define a rectangle
dst = np.float32([[0, imgsz],  # Bottom left
                  [imgsz, imgsz],  # Bottom right
                  [imgsz, 0],  # Top right
                  [0, 0]])  # Top left

points = []

def select_point(event, x, y, flags, param):
    global points, imgsz

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(im0, (x, y), 10, (0, 255, 0), -1)
        points.append([x, y])
        print(f'Point selected: ({x},{y})')  # Print the selected point's coordinates

        if len(points) == 4:
            src = np.float32(points)
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(im0, M, (imgsz, imgsz))
            cv2.imshow('Warped Image', warped)

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', select_point)

cv2.imshow('Image', im0)
cv2.waitKey(0)
cv2.destroyAllWindows()
