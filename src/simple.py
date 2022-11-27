import cv2
import numpy as np

path = '/images/shape1.png'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
_, threshold = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

cv2.imshow("shapes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()