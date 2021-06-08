# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 19:49:38 2021

@author: Julian
"""
import cv2
import numpy as np
import pyttsx3
import matplotlib.pyplot as plt

engine = pyttsx3.init()

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('shot4.mp4')
# tracker = tr.EuclideanDistTracker()
# Read until video is completed
trajectory = traj = np.empty((1, 2), int)
cont = 0
release_angle = -1
scrn_fract = 3


while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame1 = cap.read()
  ret, frame2 = cap.read()

  
  if ret == True:
      
    height = frame1.shape[0]
    width  = frame1.shape[1]
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)     
    dilated = cv2.dilate(thresh, (3,3), iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        
    #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
    
    cont_prev = cont
    cont = 0
    origin = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300:
            # cv2.drawContours(frame1, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            if y < (height/scrn_fract):
                cont = cont + 1
                centerx = int(x + w/2)
                centery = int(y + h/2)
                centery_aux = height-int(y + h/2)
                cv2.rectangle(frame1, (x,y), (x+w, y+h), (0,255,0), 3)
                trajectory = np.append(trajectory, np.array([[centerx,centery]]), axis=0)
                traj  = np.append(traj, np.array([[centerx,centery_aux]]), axis=0)
                j = traj.shape[0]-1
                angle = np.rad2deg(np.arctan2(traj[j,1] - traj[j-2,1], traj[j,0] - traj[j-2,0]))
                angle = int(angle)
                
                # frame1 = cv2.putText(frame1, str(angle), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                # print(cont_prev, cont, angle)
                
                
    print (cont_prev, cont)
    if cont == 2 and cont_prev == 1 and release_angle == -1:
        if angle < 90: release_angle = angle
        else: release_angle = 180 - angle
        engine.say(str(release_angle))
        engine.runAndWait()
        
    
    if cont == 0 and cont_prev == 1:
        trajectory = traj = np.empty((1, 2), int)
        release_angle = -1
        
    
    if release_angle >= 0:    
      frame1 = cv2.putText(frame1, 'RELEASE ANGLE = ' + str(release_angle), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    
    for i in range(1, trajectory.shape[0]-1):
        cv2.line(frame1, (trajectory[i,0],trajectory[i,1]), (trajectory[i+1,0], trajectory[i+1,1]), (0,255,0), 3)
    #print(number)
    
    
    cv2.imshow('Frame',frame1)
    cv2.imshow('Threshold', thresh)

    # Press Q on keyboard to  exit
    key = cv2.waitKey(30)
    if key == ord('q'):
        break
    if key == ord('p'):
        cv2.waitKey(-1) #wait until any key is pressed
  # Break the loop
  else: 
    break


# When everything done, release the video capture object
# traj = np.delete(traj,0, axis=0)
# plt.plot(traj[:,0], traj[:,1], c='r' )
# angle_arr = np.empty((1, 0), int)
# for i in range(1, traj.shape[0]-1):
#     # print(traj[i,1]-traj[i+1,1],traj[i,0]-traj[i+1,0])
#     angle = np.rad2deg(np.arctan2(traj[i+1,1] - traj[i,1], traj[i+1,0] - traj[i,0]))
#     # angle = np.rad2deg(np.arctan2(y[-1] - y[0], x[-1] - x[0]))
#     angle_arr = np.append(angle_arr, angle)
    
# plt.plot(traj[2:,0], angle_arr, c='r' )
engine.stop()
cap.release()
# Closes all the frames
cv2.destroyAllWindows()
