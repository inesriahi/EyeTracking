import math
import numpy as np
import dlib
import cv2

margin = 5

def midpoint(p1, p2):
    '''
    Returns the coordinates of the midpoint between the two given points from the landmarks variable

    :param p1: point 1
    :param p2: point 2
    :return: tuple containing x and y coordinates of the middle point
    '''
    return (p1.x+p2.x)//2, (p1.y+p2.y)//2

def distance(p1,p2):
    '''
    Returns the distance between two points represented as tuples

    :param tuple p1: point 1
    :param tuple p2: point 2
    :return: the euclidean distance between the 2 points
    '''
    # calculate distance between x coordinates
    x_diff = p1[0] - p2[0]
    # calculate distance between y coordinates
    y_diff = p1[1] - p2[1]
    return np.sqrt(x_diff*x_diff + y_diff*y_diff)

def get_blinking_ratio(eye_points, facial_landmarks):
    '''
    Returns the ratio between the eye's horizontal line and the vertical line

    :param eye_points: list containing the points representing the eye landmarks according to dlib
    :param facial_landmarks: landmarks of the face by dlib
    :return: blinking ratio: the ratio between the eye's horizontal line and the vertical line
    '''
    # get eye's left point and right point coordinates
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    # calculate the eye's top center and bottom center using midpoint() function
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    # compute the eye's vertical and the horizontal line length using distance() function
    ver_line_length = distance(center_top, center_bottom)
    hor_line_length = distance(left_point, right_point)
    # get the blink ratio by dividing eye's horizontal line by the vertical line and return it
    ratio = hor_line_length / ver_line_length
    return ratio

def process_eye(eye_frame, threshold):
    '''
    Process the eye to get the thresholded version of it

    :param eye_frame: Frame containing the segmented eye using polyfill
    :param threshold: Value for binary thresholding
    :return: Binary thresholded eye with the same size of :param eye_frame
    '''
    # Enlarge the eye size
    eye = cv2.resize(eye_frame, None, fx=5, fy=5)
    # Threshold the Eye to get the iris in black with white background
    _, eye = cv2.threshold(eye, threshold, 255, cv2.THRESH_BINARY)

    # Get an ellipse SE
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    # Perform Closing operation to close small holes
    closing = cv2.morphologyEx(eye, cv2.MORPH_CLOSE, kernel)

    # Get the original eye size by reducing its size again
    small_eye = cv2.resize(closing, None, fx=1 / 5, fy=1 / 5)

    return small_eye


def iris_size(eye_frame):
    '''
    Returns the iris size: the ratio between nonzero pixels and the total size of the frame

    :param eye_frame: frame containing only the binary thresholded eye
    :return: number of nonzero pixels divided by the size of the frame
    '''
    # get original eye frame without margin
    frame = eye_frame[margin:-margin,margin:-margin]
    h, w = frame.shape[:2]
    # calculate the iris size by getting the ratio between nonzero pixels (iris pixels) and the whole frame pixels
    n_pixels = h * w
    n_blocks = n_pixels - cv2.countNonZero(frame)
    return n_blocks / n_pixels


def best_threshold(eye_frame):
    '''
    Find the best threshold for the iris evaluated its average size

    :param eye_frame: Frame containing only the eye
    :return: best found threshold to get the iris
    '''
    # set average iris size (got by experimenting)
    average_iris_size = 0.30
    # trials array to choose the best from them
    trials = {}

    # loop through threshold values and apply it to eye frame
    for threshold in range(5,100,5):
        iris_frame = process_eye(eye_frame, threshold)
        trials[threshold] = iris_size(iris_frame)

    # get the best threshold from all trials
    best_thresh, best_iris_size = min(trials.items(), key=(lambda p: abs(p[1]-average_iris_size)))

    return best_thresh

def get_eye_center(landmarks):
    '''
    Returns the left eye center

    :param landmarks: face landmarks
    :return: the x and y coordinates of the eye center
    '''
    ## TODO: Find better get center logic

    # get eye center location using estimations from dlib points
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
    Return the pupil angle direction and distance from the center of the eye.

    :param landmarks: The landmarks of the face
    :param pupil: The coordinates of the Pupil
    :return: angle in degrees and distance
    '''
    # get eye center
    eyeCenter = get_eye_center(landmarks)
    # compute gaze direction by calculating the angle and the distance between the pupil and eye center
    angle = math.degrees(math.atan2(eyeCenter[1]-pupil[1], eyeCenter[0]-pupil[0]))
    dist = distance(pupil,eyeCenter)

    return angle, dist

def get_direction_description(detected_angle, detected_distance):
    '''
    Returns text description for the direction of looking

    :param detected_angle: angle in degrees between eye center and pupil
    :param detected_distance: distance between eye center and pupil
    :return: text description for the position of the pupil
    '''
    # if the distance between pupil and eye center is less than a specific value,
    # the gaze is estimated to be at the center
    if detected_distance < 3.5:
        return "center"
    # dictionary that stores the directions of pupil with their angle from eye center
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
    # compute the best estimation of pupil direction, by finding the minimum difference
    # between the detected angle and the angle of each direction (from the directions dictionary)
    dir_name, nearest_anchor = min(directions.items(), key=(lambda a: abs(detected_angle - a[1])))

    return dir_name
