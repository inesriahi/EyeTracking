import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from helpers import *

dir = 'test_imgs'
# initialize dlib face detector
detector = dlib.get_frontal_face_detector()
# initialize facial landmark predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

all_files = sorted(os.listdir(dir))

count = 20
errors = []

for i in range(len(all_files)):
    file_dir = ''.join([dir, '/', all_files[i]])
    if file_dir.endswith('.eye'):
        file_dir_eye = ''.join([dir, '/', all_files[i]])
        file_dir_pgm = ''.join([dir, '/', all_files[i + 1]])
        with open(file_dir_eye, 'r') as coor_file:
            coor_file.readline()  # skip header
            x, y = map(int, coor_file.readline().split()[2:])
            real_pupil = (x, y)
        frame = cv2.imread(file_dir_pgm)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # rotate the frame horizontally
        # frame = cv2.flip(frame, 1)
        # gray = cv2.flip(gray, 1)

        # array of all the faces
        faces = detector(gray)
        # loop over the faces frames
        found = True
        for face in faces:
            # get face frame corners
            x, y = face.left(), face.top()
            x1, y1 = face.right(), face.bottom()
            # get facial landmarks for the face
            landmarks = predictor(gray, face)

            # Gaze Detection

            # get left eye points
            left_eye_region = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)],
                                       np.int32)
            # create a black frame with frame size
            black_frame = np.zeros(gray.shape, np.uint8)
            # create a mask to extract the left eye
            mask = np.full(gray.shape, 255, np.uint8)
            # cv2.polylines(mask,[left_eye_region], True, 255)
            # draw mask of left eye area
            cv2.fillPoly(mask, [left_eye_region], (0, 0, 0))
            # get the cropped area of the mask
            left_eye_masked = cv2.bitwise_not(black_frame, gray.copy(), mask=mask)
            # cv2.imshow("", left_eye_masked)

            # get eye detection rectangle coordinates
            min_x = np.min(left_eye_region[:, 0]) - margin
            max_x = np.max(left_eye_region[:, 0]) + margin
            min_y = np.min(left_eye_region[:, 1]) - margin
            max_y = np.max(left_eye_region[:, 1]) + margin

            ## Eye
            # Get the only the eye from the masked frame (with eye and white background)
            small_eye = left_eye_masked[min_y:max_y, min_x:max_x]
            # cv2.imshow("Left Eye Masked Small Eye", small_eye)
            # Process the eye with the best threshold
            small_eye = process_eye(small_eye, best_threshold(small_eye))

            # Show the thresholded iris area in its original size
            # cv2.imshow("Small Eye resized", small_eye)

            ## Frame reconstruction to replace the thresholded iris in its frame
            white_frame = np.full(gray.shape, 255, np.uint8)
            white_frame[min_y:max_y, min_x:max_x] = small_eye
            # cv2.imshow("Reconstructed Frame", white_frame)

            # Invert the image to make the iris white as it is the object
            black_bg = cv2.bitwise_not(white_frame)
            ## Finding Contours
            contours, _ = cv2.findContours(black_bg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)

            # skip closed eyes images
            if len(contours) == 0:
                found = False
                break
            ## for each contour
            # compute the center of the contour
            for cnt in contours:
                # Get the image moments to get the center of the image
                M = cv2.moments(cnt)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Center of the iris is the pupil
                detected_pupil = (cX, cY)
                # draw the contour and center of the shape on the image

                # cv2.drawContours(black_bg, [cnt], -1, (0, 255, 0), 2)
                # cv2.circle(frame, (cX, cY), 2, (0, 255, 0), -1)
                # cv2.putText(frame, f'Coordinates of pupil: ({cX},{cY})', (30, 50), 0, 0.5, (0, 0, 0))
                break

        if not found:
            continue

        # cv2.circle(frame, real_pupil, 2, (0, 0, 255), -1)

        # cv2.circle(frame, (x_left, y_left),2, (0,255,0),1)
        # frame = cv2.flip(frame, 1)
        # frame = cv2.resize(frame, None, fx=3,fy=3)

        # cv2.imshow(all_files[i + 1], frame)

        # count -= 1
        # if count == 0:
        #     break
        errors.append(distance(detected_pupil, real_pupil))

errors = np.array(errors)
print('Number of images:',errors.shape)
print('Max error:', errors.max())
print('Min error:', errors.min())
print('Quantiles:', np.quantile(errors, [0.25,0.5,.75]))
hist = sns.histplot(errors, color='salmon')
hist.set(xlim=(0,20))
plt.xlabel('Error (Euclidean distance between real and predicted pupil)')
# plt.xticks(range(0,5))
# plt.xlim(0,6)
plt.show()
# Calculating the error
# print(all_files[i + 1] + ': Detected Pupil:', detected_pupil, 'Real Pupil:', real_pupil)


cv2.waitKey(0)
cv2.destroyAllWindows()
