import math

import numpy as np
import dlib
import cv2


cap = cv2.VideoCapture(0)

margin = 5
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
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
    Get the Eye
    :param landmarks: The landmarks of the face
    :param pupil: The coordinates of the Pupil
    :return: angle in degrees and distance
    '''
    eyeCenter = get_eye_center(landmarks)
    angle = math.degrees(math.atan2(eyeCenter[1]-pupil[1], eyeCenter[0]-pupil[0]))
    dist = distance(pupil,eyeCenter)

    return angle, dist


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret:
        faces = detector(gray) # array of all the faces
        for face in faces:
            x,y = face.left(), face.top()
            x1, y1 = face.right(), face.bottom()

            landmarks = predictor(gray, face)

            # Gaze Detection
            left_eye_region = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36,42)], np.int32)

            black_frame = np.zeros(gray.shape, np.uint8)

            mask = np.full(gray.shape, 255, np.uint8)
            # cv2.polylines(mask,[left_eye_region], True, 255)
            cv2.fillPoly(mask, [left_eye_region], (0,0,0))
            left_eye_masked = cv2.bitwise_not(black_frame, gray.copy(),mask=mask)
            # cv2.imshow("", left_eye_masked)

            min_x = np.min(left_eye_region[:, 0]) - margin
            max_x = np.max(left_eye_region[:, 0]) + margin
            min_y = np.min(left_eye_region[:, 1]) - margin
            max_y = np.max(left_eye_region[:, 1]) + margin

            ## Eye
            # Get the only the eye from the masked frame (with eye and white background)
            small_eye = left_eye_masked[min_y:max_y, min_x:max_x]
            cv2.imshow("Left Eye Masked Small Eye", small_eye)

            small_eye = process_eye(small_eye, best_threshold(small_eye))

            # Show the thresholded iris area in its original size
            cv2.imshow("Small Eye resized", small_eye)

            ## Frame reconstruction to replace the thresholded iris in its frame
            white_frame = np.full(gray.shape, 255, np.uint8)
            white_frame[min_y:max_y, min_x:max_x] = small_eye
            cv2.imshow("Reconstructed Frame", white_frame)

            # Invert the image to make the iris white as it is the object
            black_bg = cv2.bitwise_not(white_frame)
            ## Finding Contours
            contours, _ = cv2.findContours(black_bg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)

            ## for each contour

            # compute the center of the contour
            for cnt in contours:
                M = cv2.moments(cnt)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                pupil = (cX, cY)
                #draw the contour and center of the shape on the image
                cv2.drawContours(black_bg, [cnt], -1, (0,255,0), 2)
                cv2.circle(frame, (cX, cY), 2, (0,255,0),-1)
                cv2.putText(frame, f'Coordinates of pupil: ({cX},{cY})', (30,50), 0,0.5,(0,0,0))
                break


            ### Detect Blinking
            left_eye_ratio = get_blinking_ratio(list(range(36,42)), landmarks)
            right_eye_ratio = get_blinking_ratio(list(range(42,48)), landmarks)
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

            if blinking_ratio > 5.5:
                cv2.putText(frame, "Blinking", (30,100), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,0,0))


            eye_center = get_eye_center(landmarks)
            cv2.circle(frame, eye_center, 2, (0, 0, 255), -1)

            # ##### Eye direction distance
            angle, dist = eye_direction_distance(landmarks, pupil)
            cv2.putText(frame, f'angle: {angle:.4f} & distance: {dist:.4f}', (30, 150), 0, 0.5, (0, 0, 0))
            cv2.imshow('Black Bg', black_bg)

        cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()