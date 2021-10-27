# THIS CODE IS NOT FUNCTIONAL, it is legacy code from tests made in a CUI environment
# to test the feasibility of the detection of meniscus in a test tube or burette
# Now the GUI version is the working one.
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2
import numpy as np
from Paths import paths, files
from Meniscus_Utils import *
from Camera_Utils import set_cam_params, get_cam_calibration

# Load category index
category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-21')).expect_partial()


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections_ = detection_model.postprocess(prediction_dict, shapes)
    return detections_


# --- PROGRAM CONSTANTS ---
WINDOW_SIZE_FACTOR = 0.3

RES_X, RES_Y = int(1920), int(1080)  # int(800), int(600) / int(1280), int(720) / int(1920), int(1080)
BRIGHTNESS = 120
FOCUS = 35
ROI_PERCENTAGE_X = 100
ROI_PERCENTAGE_Y = 50
CALIBRATION_FILE = "camera_calibration_file.txt"
USE_CALIBRATION = False

INVERTED_SCALE = True               # If minimum value of the scale is on top use True (usually for test tubes)
CAPACITY = [0, 10]                  # Capacity marks in cm3 defined as min value and max value

FONT_SIZE = 2                       # Factor size in image (MUST BE INTEGER)
LINE_WIDTH = 2                      # Lines width



x1 = int((RES_Y - (RES_Y * ROI_PERCENTAGE_Y / 100)) / 2)
x2 = int(x1 + (RES_Y * ROI_PERCENTAGE_Y / 100))
y1 = int((RES_X - (RES_X * ROI_PERCENTAGE_X / 100)) / 2)
y2 = int(y1 + (RES_X * ROI_PERCENTAGE_X / 100))
print('ROI: ', x1, x2, y1, y2)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
w, h, b, f, af = set_cam_params(cap, RES_X, RES_Y, BRIGHTNESS, FOCUS, False)
w, h = int(float(w)), int(float(h))
print('Resolution(w,h): ', w, h, ' / Brightness: ', b, ' / Focus: ', f, ' / Autofocus: ', af)
cv2.namedWindow('Image')
static_mark = Mark(CAPACITY, WINDOW_SIZE_FACTOR, INVERTED_SCALE)
meniscus = Meniscus()
cv2.setMouseCallback('Image', static_mark.mark_pos)
# mtx, new_mtx, dist, roi = get_cam_calibration(CALIBRATION_FILE, w, h)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # if USE_CALIBRATION:
        #     frame = cv2.undistort(frame, mtx, dist, None, new_mtx)
        image_np = np.array(frame)
        height = y2 - y1
        width = x2 - x1
        image_np = image_np[y1:y2, x1:x2]
        image_np_with_detections = image_np.copy()

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections
        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_np_with_detections, meniscus = meniscus_draw(image_np_with_detections,
                                                           detections['detection_boxes'],
                                                           detections['detection_scores'],
                                                           static_mark,
                                                           max_boxes=2,
                                                           min_score_thresh=0.2)

        draw_levels(image_np_with_detections, meniscus, LINE_WIDTH, FONT_SIZE)      # Draw line and text at the meniscus lower edge
        draw_center_lines(image_np_with_detections, LINE_WIDTH)                     # Draw centered reference lines
        draw_marks(image_np_with_detections, static_mark, LINE_WIDTH, FONT_SIZE)    # Draw the marks that limit the volume calculation
        # print('Levels  = ', meniscus.yposition)
        print('Volumes = ', meniscus.reading)
        print('Marks   = ', static_mark.yposition)
        # print_img_resolution(image_np_with_detections)
        cv2.imshow('Image', cv2.resize(image_np_with_detections,
                                       (int(width * WINDOW_SIZE_FACTOR), int(height * WINDOW_SIZE_FACTOR))))
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
