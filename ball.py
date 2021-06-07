# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 19:49:38 2021

@author: Julian
"""
import cv2
from tracker import *
import numpy as np
import matplotlib.pyplot as plt

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('shot.mp4')
# tracker = tr.EuclideanDistTracker()
# Read until video is completed
trajectory = traj = np.empty((1, 2), int)

while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame1 = cap.read()
  ret, frame2 = cap.read()
  height = frame1.shape[0]
  
  if ret == True:
   
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)     
        dilated = cv2.dilate(thresh, (3,3), iterations=3)
        _, contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        
        #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
        number=0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 400:
                # cv2.drawContours(frame1, [cnt], -1, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(cnt)
                centerx = int(x + w/2)
                centery = int(y + h/2)
                centery_aux = height-int(y + h/2)
                cv2.rectangle(frame1, (x,y), (x+w, y+h), (0,255,0), 3)
                number=number+1
                print(number)
                trajectory = np.append(trajectory, np.array([[centerx,centery]]), axis=0)
                traj  = np.append(traj, np.array([[centerx,centery_aux]]), axis=0)
        
        for i in range(1, trajectory.shape[0]-1):
            cv2.line(frame1, (trajectory[i,0],trajectory[i,1]), (trajectory[i+1,0], trajectory[i+1,1]), (0,255,0), 3)
        #print(number)
        
        cv2.imshow('Frame',frame1)
        cv2.imshow('Threshold', thresh)

        # Press Q on keyboard to  exit
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
traj = np.delete(traj,0, axis=0)
plt.plot(traj[:,0], traj[:,1], c='r' )

cap.release()

# Closes all the frames
cv2.destroyAllWindows()
