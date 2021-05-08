# This is a sample Python script.

import cv2
import numpy as np
import dlib
import itertools

from math import hypot
kBlurSize = 5
cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def lookup_displacement(ref, dimension):
    return displace_table[:,
      (displace_center[0] - ref[0]) : (displace_center[0] + dimension[0] - ref[0]),
      (displace_center[1] - ref[1]) : (displace_center[1] + dimension[1] - ref[1])]

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

def gradientX(img):
    return np.gradient(img)[::-1]


def computeDynamicThreshold(gradientMatrix, stdDevFactor):
    meanMagnGrad, stdMagnGrad = cv2.meanStdDev(gradientMatrix)
    stdDev = stdMagnGrad[0] / np.sqrt(gradientMatrix.shape[0] * gradientMatrix.shape[1])
    # print(meanMagnGrad, stdMagnGrad)
    return stdDevFactor * stdDev + meanMagnGrad[0]


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray) # array of all the faces
    for face in faces:
        x,y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()


        landmarks = predictor(gray, face)

        # Detect blinking
        # left_eye_ratio = get_blinking_ratio(list(range(36,42)), landmarks)
        # right_eye_ratio = get_blinking_ratio(list(range(42,48)), landmarks)
        # blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        #
        # if blinking_ratio > 5:
        #     cv2.putText(frame, "Blinking", (50,150), cv2.FONT_HERSHEY_DUPLEX, 3, (255,0,0))

        # Gaze Detection
        left_eye_region = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36,42)], np.int32)
        # cv2.polylines(frame, [left_eye_region], True, (0,0,255), 2)

        min_x = np.min(left_eye_region[:,0])
        max_x = np.max(left_eye_region[:,0])
        min_y = np.min(left_eye_region[:,1])
        max_y = np.max(left_eye_region[:,1])

        eye = gray[min_y:max_y, min_x:max_x]
        eye = cv2.resize(eye, None, fx=5, fy=5)

        # https://github.com/trishume/eyeLike/blob/78df488c0b0d00ee58aa1c6eaa52e9ad0de11ae4/src/findEyeCenter.cpp#L112
        grad_x = gradientX(eye)[0]
        grad_y = np.transpose(gradientX(np.transpose(eye))[0])

        # Compute all the magnitudes
        magnitudes = cv2.magnitude(grad_x, grad_y)
        # Compute the Threshold
        gradient_threshold = computeDynamicThreshold(magnitudes,50)
        #Normalize
        # print(eye.shape)

        magnitudes_mask = magnitudes > gradient_threshold
        magnitudes_masked = magnitudes_mask * magnitudes
        grad_x = np.divide(grad_x, magnitudes_masked, out=np.zeros_like(grad_x), where=magnitudes_masked!=0) #when nan replace with zero
        grad_y = np.divide(grad_y, magnitudes_masked, out=np.zeros_like(grad_y), where=magnitudes_masked!=0) #when nan replace with zero

        weight = cv2.GaussianBlur(eye, (kBlurSize, kBlurSize), 0)
        weight = np.invert(weight)

        gradient = np.stack((grad_x, grad_y),0)


        ### Dispalcement
        displace_center = np.array((100, 100))
        displace_table = np.indices(eye.shape) - displace_center[:, None, None]
        displace_table = displace_table[::-1, :, :] / (np.linalg.norm(displace_table[::-1, :, :], 2, 0, True) + 1e-10)

        d = lookup_displacement((0,0), eye.shape)
        print(f'dshape {d.shape}')
        t = np.zeros_like(eye)
        # for c in itertools.product(range(eye.shape[0]), range(eye.shape[1])):
        #     c = np.array(c)
        #     d = lookup_displacement(c, eye.shape)
        #     print(d)
        #     s = np.sum(d*gradient,0)
        #     t[c[0], c[1]] = np.mean(np.maximum(0, s)**2)
        #
        # result = np.unravel_index(np.argmax(t), t.shape)

        # print(displace_table)

        # for y in range(eye.shape[0]):
        #     Xr = grad_x[y]
        #     Yr = grad_y[y]
        #     print(Yr)
        #     Mr = magnitudes[y]
        #     for x in range(eye.shape[1]):
        #         gX = Xr[x]
        #         gY = Yr[x]
        #         magnitude = Mr[x]
        #         if (magnitude > gradient_threshold):
        #             Xr[x] = gX/magnitude
        #             Yr[x] = gY/magnitude
        #         else:
        #             Xr[x] = 0
        #             Yr = 0

        cv2.imshow("Weight", weight)



        # print(magnitude)
        cv2.imshow("Eye", eye)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()