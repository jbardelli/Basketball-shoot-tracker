# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 11:56:43 2021

@author: jbardelli
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 19:49:38 2021

@author: Julian
"""
import cv2


tracker = cv2.TrackerMedianFlow_create()

cap = cv2.VideoCapture('shot.mp4')

ok, frame = cap.read()
print('OK1')
# bbox = (300, 23, 86, 320)
bbox = cv2.selectROI(frame, False)
ok = tracker.init(frame, bbox)
print('OK2')

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()

  if ret == True:
        ok, bbox = tracker.update(frame)
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

          
    
        cv2.imshow('Frame',frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
