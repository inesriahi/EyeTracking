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
        # cv2.imshow("", left_eye_masked)

        margin = 5

        min_x = np.min(left_eye_region[:, 0]) - margin
        max_x = np.max(left_eye_region[:, 0]) + margin
        min_y = np.min(left_eye_region[:, 1]) - margin
        max_y = np.max(left_eye_region[:, 1]) + margin

        #
        # kernel = np.ones((3,3), np.uint8)
        # iris_frame = cv2.bilateralFilter(left_eye_masked, 10,15,15)
        # iris_frame = cv2.erode(iris_frame, kernel, iterations=3)
        # _,iris_frame = cv2.threshold(iris_frame, 70,255,cv2.THRESH_BINARY)
        # cv2.imshow("Iris Frame", iris_frame)

        ## Eye
        # Get the only the eye from the masked frame (with eye and white background)
        small_eye = left_eye_masked[min_y:max_y, min_x:max_x]
        cv2.imshow("Left Eye Masked Small Eye", small_eye)

        # Enlarge the eye size
        eye = cv2.resize(small_eye, None, fx=5, fy=5)
        # Threshold the Eye to get the iris in black with white background
        _,eye = cv2.threshold(eye, 70,255,cv2.THRESH_BINARY)

        # Get an ellipse SE
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
        # Perform Closing operation to close small holes
        closing = cv2.morphologyEx(eye, cv2.MORPH_CLOSE, kernel)

        # Get the original eye size by reducing its size
        small_eye = cv2.resize(closing, None, fx=1/5, fy=1/5)
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
        for cnt in contours:
            # compute the center of the contour
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            #draw the contour and center of the shape on the image
            cv2.drawContours(black_bg, [cnt], -1, (0,255,0), 2)
            cv2.circle(frame, (cX, cY), 2, (0,255,0),-1)


        cv2.imshow('Black Bg', black_bg)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()