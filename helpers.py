import math
import numpy as np
import dlib
import cv2

margin = 5

def midpoint(p1, p2):
    return (p1.x+p2.x)//2, (p1.y+p2.y)//2

def distance(p1,p2):
    x_diff = p1[0] - p2[0]
    y_diff = p1[1] - p2[1]
    return np.sqrt(x_diff*x_diff + y_diff*y_diff)

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)

    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    # hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    # ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    ver_line_length = distance(center_top, center_bottom)
    hor_line_length = distance(left_point, right_point)

    ratio = hor_line_length / ver_line_length
    return ratio

def process_eye(eye_frame, threshold):
    # Enlarge the eye size
    eye = cv2.resize(eye_frame, None, fx=5, fy=5)
    # Threshold the Eye to get the iris in black with white background
    _, eye = cv2.threshold(eye, threshold, 255, cv2.THRESH_BINARY)

    # Get an ellipse SE
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    # Perform Closing operation to close small holes
    closing = cv2.morphologyEx(eye, cv2.MORPH_CLOSE, kernel)

    # Get the original eye size by reducing its size
    small_eye = cv2.resize(closing, None, fx=1 / 5, fy=1 / 5)

    return small_eye


def iris_size(eye_frame):
    frame = eye_frame[margin:-margin,margin:-margin]
    h, w = frame.shape[:2]
    n_pixels = h * w
    n_blocks = n_pixels - cv2.countNonZero(frame)
    return n_blocks / n_pixels


def best_threshold(eye_frame):
    average_iris_size = 0.30
    trials = {}

    for threshold in range(5,100,5):
        iris_frame = process_eye(eye_frame, threshold)
        trials[threshold] = iris_size(iris_frame)

    best_thresh, best_iris_size = min(trials.items(), key=(lambda p: abs(p[1]-average_iris_size)))

    return best_thresh

def get_eye_center(landmarks):
    ## TODO: Find better get center logic
    mid_eye_x = (landmarks.part(37).x + landmarks.part(38).x) // 2

    # mid_eye_x = (landmarks.part(36).x + landmarks.part(39).x)//2

    # mid_upper = (landmarks.part(37).y + landmarks.part(38).y) // 2
    # mid_lower = (landmarks.part(40).y + landmarks.part(41).y) // 2
    # mid_eye_y = (mid_upper + mid_lower) // 2

    # mid_eye_y = landmarks.part(27).y

    mid_eye_y = landmarks.part(36).y
    center = (mid_eye_x, mid_eye_y)

    return center

def eye_direction_distance(landmarks, pupil):
    '''
    Return the pupil angle direction and distance from the center of the eye
    :param landmarks: The landmarks of the face
    :param pupil: The coordinates of the Pupil
    :return: angle in degrees and distance
    '''
    eyeCenter = get_eye_center(landmarks)
    angle = math.degrees(math.atan2(eyeCenter[1]-pupil[1], eyeCenter[0]-pupil[0]))
    dist = distance(pupil,eyeCenter)

    return angle, dist

def get_direction_description(detected_angle, detected_distance):
    if detected_distance < 3.5:
        return "center"
    directions = {
        "top" : 90,
        "bottom" : -90,
        "right" : 180,
        "left" : 0,
        "top_right" : 135,
        "bottom_right" : -135,
        "top_left" :  45,
        "bottom_left" : -45
    }

    dir_name, nearest_anchor = min(directions.items(), key=(lambda a: abs(detected_angle - a[1])))
    return dir_name
