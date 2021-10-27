import cv2
import uuid
import os
from Paths import IMAGES_PATH
from Select_ROI_on_Video import staticROI

def set_res(cap, x,y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    return str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

labels = ['meniscus']
number_imgs = 10

# Static_ROI = staticROI(0, 800, 600) # Define StaticROI class variable Cam_number, width_pixels, height
# roi = Static_ROI.update()           # Call to select ROI specify webcam number
# roi = [(150,0), (450, 800)]
roi = []
print('ROI = ',roi)

if not os.path.exists(IMAGES_PATH):
    if os.name == 'nt':
        print(IMAGES_PATH)
        os.makedirs(IMAGES_PATH)

for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        os.mkdir(path)

WINDOW_SIZE_FACTOR = 0.5
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
w, h = set_res(cap, 1280, 720)
w = int(float(w))
h = int(float(h))
print(w, h)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, 20)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 110)

for label in labels:
    print('Collecting images for {}'.format(label))
    print('Number of images to take', number_imgs)

    for imgnum in range(number_imgs):
        print('Collecting image {}, press t when ready'.format(imgnum))
        while True:
            ret, frame1 = cap.read()
            frame1 = cv2.rotate(frame1, cv2.ROTATE_90_CLOCKWISE)
            if ret:
                if not roi:
                    frame = frame1
                else:
                    frame = frame1[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
                cv2.imshow('Frame', cv2.resize(frame, (int(h * WINDOW_SIZE_FACTOR), int(w * WINDOW_SIZE_FACTOR))))
                key = cv2.waitKey(1)
                if key == ord('t') or key == ord('q'):
                    break
        if key == ord('q'):
            break
        print('Image {} taken'.format(imgnum))
        imgname = os.path.join(IMAGES_PATH, label, label + '.' +'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imgname, frame)

cap.release()
cv2.destroyAllWindows()


