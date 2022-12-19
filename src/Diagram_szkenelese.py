import cv2
import numpy as np

def preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (11, 11), 1)
    img_canny = cv2.Canny(img_blur, 200, 0)
    kernel = np.ones((5, 5))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=1)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    return img_erode

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    foundObject = set()
    circleCount = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 250 < area < 10000:
            cv2.drawContours(imgContour, cnt, -1, (155, 155, 0), 1)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            hull = cv2.convexHull(approx, returnPoints=False)
            sides = len(hull)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            objectType = "None"
            if objCor == 4:
                aspRatio = w / float(h)
                if 0.98 < aspRatio < 1.03:
                    objectType = "Square"
                else:
                    objectType = "Rectangle"
            elif objCor > 7:
                objectType = "Circle"
                circleCount = circleCount + 1
            foundObject.add(objectType)
            cv2.putText(imgContour, objectType, (x + (w // 2) - 10, y + (h // 2) - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
        else:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.putText(imgContour, "Arrow", (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 0, 0), 1)
            foundObject.add("Arrow")
    requiredObjects = {"Arrow", "Circle", "Rectangle"}
    if (requiredObjects.issubset(foundObject)) and (circleCount > 1):
        cv2.putText(imgContour, "Valid Diagram", (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                    (0, 0, 0), 1)
    else:
        cv2.putText(imgContour, "Not Valid Diagram", (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                    (0, 0, 0), 1)

path = 'images/002_test.jpg'
img = cv2.imread(path)
imgContour = img.copy()

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
imgCanny = cv2.Canny(imgBlur, 50, 50)
getContours(preprocess(img))

imgBlank = np.zeros_like(img)
imgStack = stackImages(0.7, ([img, imgContour]))

cv2.imshow("Stack", imgStack)

cv2.waitKey(0)
