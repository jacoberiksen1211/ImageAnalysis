# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 22:09:10 2022

@author: Jacob
"""

import time
import cv2
import numpy as np


def show_in_moved_window(win_name, img, x, y):
    """
    Show an image in a window, where the position of the window can be given
    """
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name,img)


def capture_from_camera_and_show_images():
    print("Starting image capture")
    
    #Set values needed for processing
    threshold = 10 #threshold to create binary from dif image
    alpha = 0.95 #value to keep from acumulated background
    alertVal = 0.05 #limit of percantage of changed pixels before raising alarm

    print("Opening connection to camera")
    url = 0
    use_droid_cam = False
    if use_droid_cam:
        url = "http://192.168.1.120:4747/video"
    cap = cv2.VideoCapture(url)
    # cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Starting camera loop")
    # Get first image
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame")
        exit()

    # Transform image to gray scale and then to float, so we can do some processing
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    #NEW:
    print(frame_gray.shape)
    print(f"total pixels in frame = {480*640}")
    total_pixels = 480*640

    # To keep track of frames per second
    start_time = time.time()
    n_frames = 0
    stop = False
    while not stop:
        ret, new_frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # Transform image to gray scale and then to float, so we can do some processing
        new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Compute difference image
        dif_img = np.abs(new_frame_gray - frame_gray)
        
        #create binary image based on threshold of difference
        bin_img = dif_img > threshold
        
        #count number of changed pixels
        F = np.sum(bin_img)
        
        #compute percentage
        percentage = F / total_pixels
        
        #if alarm then show alarm
        if(percentage > alertVal):
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(new_frame, "ALERT", (10, 50), font, 1, [0, 0, 255], 1)

        # Keep track of frames-per-second (FPS)
        n_frames = n_frames + 1
        elapsed_time = time.time() - start_time
        fps = int(n_frames / elapsed_time)

        # Put the FPS on the new_frame
        str_out = f"fps: {fps}"
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(new_frame, str_out, (100, 100), font, 1, 255, 1)

        # Display the resulting frame
        show_in_moved_window('Input', new_frame, 0, 10)
        #show_in_moved_window('Input gray', new_frame_gray.astype(np.uint8), 600, 10)
        show_in_moved_window('Difference image', dif_img.astype(np.uint8), 1200, 10)
        #NEW
        #show difference instead of gray also show binary
        show_in_moved_window('Background image', frame_gray.astype(np.uint8), 600, 10)
        show_in_moved_window('Binary image', (bin_img*255).astype(np.uint8), 0, 540)
        
        # Old frame is updated
        # frame_gray = new_frame_gray
        
        # NEW update old frame by accumulating background 
        frame_gray = alpha * frame_gray + (1-alpha) * new_frame_gray
        
        if cv2.waitKey(1) == ord('q'):
            stop = True

    print("Stopping image loop")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture_from_camera_and_show_images()
