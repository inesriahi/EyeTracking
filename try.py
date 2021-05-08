import cv2
import numpy as np
import itertools
import matplotlib.pyplot as plt

def lookup_displacement(displace_table, ref, dimension):
    # print(displace_table.shape)
    return displace_table[:,
      (displace_center[0] - ref[0]) : (displace_center[0] + dimension[0] - ref[0]),
      (displace_center[1] - ref[1]) : (displace_center[1] + dimension[1] - ref[1])]

def gradientX(img):
    return np.gradient(img)[::-1]


def computeDynamicThreshold(gradientMatrix, stdDevFactor):
    meanMagnGrad, stdMagnGrad = cv2.meanStdDev(gradientMatrix)
    stdDev = stdMagnGrad[0] / np.sqrt(gradientMatrix.shape[0] * gradientMatrix.shape[1])
    # print(meanMagnGrad, stdMagnGrad)
    return stdDevFactor * stdDev + meanMagnGrad[0]


eye = cv2.imread('try1.png',0)


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

weight = cv2.GaussianBlur(eye, (5, 5), 0)
weight = np.invert(weight)

gradient = np.stack((grad_x, grad_y),0)


### Dispalcement
displace_center = np.array((100, 100))
displace_table = np.indices(eye.shape) - displace_center[:, None, None]

displace_table = displace_table[::-1, :, :] / (np.linalg.norm(displace_table[::-1, :, :], 2, 0, True) + 1e-10)
print('Dis table', displace_table.shape)
t = np.zeros_like(eye)
d = displace_table
for c in itertools.product(range(eye.shape[0]), range(eye.shape[1])):
    c = np.array(c)
    d = lookup_displacement(displace_table, c, eye.shape)

    t[c[0], c[1]] = np.mean(np.maximum(0, np.sum(d * gradient, 0)) ** 2)


result = np.unravel_index(np.argmax(t), t.shape)
cv2.circle(eye, result, 1, color=255)

print(result)
# plt.plot(result[0], result[1], 'bo')
plt.imshow(eye)
plt.show()
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

