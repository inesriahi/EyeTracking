from helpers import *

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# initialize dlib face detector
detector = dlib.get_frontal_face_detector()
# initialize facial landmark predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


while True:
    ret, frame = cap.read()
    # convert frame to grayscale for better prediction
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # rotate the frame horizontally
    frame = cv2.flip(frame, 1)
    gray = cv2.flip(gray, 1)
    if ret:
        # array of all the faces
        faces = detector(gray)
        # loop over the faces frames
        for face in faces:
            # get face frame corners
            x,y = face.left(), face.top()
            x1, y1 = face.right(), face.bottom()
            # get facial landmarks for the face
            landmarks = predictor(gray, face)

            # Gaze Detection

            # get left eye points
            left_eye_region = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36,42)], np.int32)
            # create a black frame with frame size
            black_frame = np.zeros(gray.shape, np.uint8)
            # create a mask to extract the left eye
            mask = np.full(gray.shape, 255, np.uint8)
            # cv2.polylines(mask,[left_eye_region], True, 255)
            # draw mask of left eye area
            cv2.fillPoly(mask, [left_eye_region], (0,0,0))
            # get the cropped area of the mask
            left_eye_masked = cv2.bitwise_not(black_frame, gray.copy(),mask=mask)
            # cv2.imshow("", left_eye_masked)

            # get eye detection rectangle coordinates
            min_x = np.min(left_eye_region[:, 0]) - margin
            max_x = np.max(left_eye_region[:, 0]) + margin
            min_y = np.min(left_eye_region[:, 1]) - margin
            max_y = np.max(left_eye_region[:, 1]) + margin

            ## Eye
            # Get the only the eye from the masked frame (with eye and white background)
            small_eye = left_eye_masked[min_y:max_y, min_x:max_x]
            cv2.imshow("Left Eye Masked Small Eye", small_eye)
            # Process the eye with the best threshold
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
                # Get the image moments to get the center of the image
                M = cv2.moments(cnt)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Center of the iris is the pupil
                pupil = (cX, cY)
                #draw the contour and center of the shape on the image
                cv2.drawContours(black_bg, [cnt], -1, (0,255,0), 2)
                cv2.circle(frame, (cX, cY), 2, (0,255,0),-1)
                cv2.putText(frame, f'Coordinates of pupil: ({cX},{cY})', (30,50), 0,0.5,(0,0,0))
                break


            ### Detect Blinking
            # calculate blinking ratio for both eyes
            left_eye_ratio = get_blinking_ratio(list(range(36,42)), landmarks)
            right_eye_ratio = get_blinking_ratio(list(range(42,48)), landmarks)
            # find the blinking ratio as the mean between the two eye blinking ratios
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2


            ## if the blinking ratio is greater than this value: detect a blink
            if blinking_ratio > 5.5:
                cv2.putText(frame, "Blinking", (30,200), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,0,0))

            # draw circle to find eye center (to make it as the origin for the iris (not very good idea)
            eye_center = get_eye_center(landmarks)
            cv2.circle(frame, eye_center, 2, (0, 0, 255), -1)

            # Get looking direction distance: (for pupil and eye center)
            angle, dist = eye_direction_distance(landmarks, pupil)
            # Display the frame with detections
            cv2.putText(frame, f'angle: {angle:.4f} & distance: {dist:.3f}', (30, 100), 0, 0.5, (0, 0, 0))
            # get the direction description in words
            cv2.putText(frame, f'Direction: {get_direction_description(angle, dist)}', (30, 150), 0, 0.5, (0, 0, 0))
            cv2.imshow('Black Bg', black_bg)

        cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()