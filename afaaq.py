import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


while True:
    ret, frame = cap.read()

    if ret is False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # cv2.rectangle(frame, (landmarks.part(36).x - 2, landmarks.part(37).y  - 5), (landmarks.part(39).x + 2, landmarks.part(40).y + 5), (0, 255, 0), 2)
        x = landmarks.part(36).x
        y = landmarks.part(37).y
        h = landmarks.part(40).y - landmarks.part(37).y
        w = landmarks.part(39).x - landmarks.part(36).x
        eyeframe = gray[y:y + h, x:x + w]
        coloreyeframe = frame[y:y + h, x:x + w]
        roi = eyeframe
        gray_roi = eyeframe
        rows, cols = roi.shape
        gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

        _, threshold = cv2.threshold(gray_roi, 30, 255, cv2.THRESH_BINARY_INV)  # THRESH_BINARY
        #    threshold = threshold.astype(np.uint8)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)

            # cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.line(roi, (x + int(w / 2), 0), (x + int(w / 2), rows), (0, 255, 0), 2)
            cv2.line(roi, (0, y + int(h / 2)), (cols, y + int(h / 2)), (0, 255, 0), 2)

            cv2.rectangle(coloreyeframe, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.line(coloreyeframe, (x + int(w / 2), 0), (x + int(w / 2), rows), (0, 255, 0), 2)
            cv2.line(coloreyeframe, (0, y + int(h / 2)), (cols, y + int(h / 2)), (0, 255, 0), 2)
            break

        cv2.imshow("Threshold", threshold)
        #        cv2.imshow("gray roi", gray_roi)
        cv2.imshow("Roi", roi)
        cv2.imshow("coloreyeframe", coloreyeframe)
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()

cv2.destroyAllWindows()