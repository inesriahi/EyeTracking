import numpy as np
import dlib
import cv2


cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

        margin = 5

        cv2.imshow("Left Eye Masked", left_eye_masked)
        min_x = np.min(left_eye_region[:, 0]) - margin
        max_x = np.max(left_eye_region[:, 0]) + margin
        min_y = np.min(left_eye_region[:, 1]) - margin
        max_y = np.max(left_eye_region[:, 1]) + margin

        kernel = np.ones((3,3), np.uint8)
        iris_frame = cv2.bilateralFilter(left_eye_masked, 10,15,15)
        iris_frame = cv2.erode(iris_frame, kernel, iterations=3)
        _,iris_frame = cv2.threshold(iris_frame, 70,255,cv2.THRESH_BINARY)
        cv2.imshow("Iris Frame", iris_frame)



        ## Eye
        eye = left_eye_masked[min_y:max_y, min_x:max_x]
        eye = cv2.resize(eye, None, fx=5, fy=5)
        _,eye = cv2.threshold(eye, 70,255,cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))
        closing = cv2.morphologyEx(eye, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("Eye", eye)



    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()