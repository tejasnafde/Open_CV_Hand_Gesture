import cv2
import numpy as np
import matplotlib.pyplot as plt



cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)

cap.open(0)
while True:
    # read image
    ret, frame = cap.read()
    if not ret:
        break

    # focusing on only hand data from the rectangle subwindow on the screen
    cv2.rectangle(frame, (300,300), (100,100), (0,255,0), 0)
    crop_img = frame[100:300, 100:300]

    # converting from BGR to grayscale
    gray = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)

    # applying gaussian blur
    value = (35,35)
    blurred = cv2.GaussianBlur(gray, value, 0)

    # thresholding using Otsu's Binarization method with all px of value greater than 127 becoming 1, and lower values becoming 0
    _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # finding all the contours 
    contours, heirarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, \
            cv2.CHAIN_APPROX_NONE)
    # extracting the contour with max area
    cnt = max(contours, key = lambda x: cv2.contourArea(x))

    # creating a bounding rectangle around the contour
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)

    # finding convex hull
    hull = cv2.convexHull(cnt)

    # drawing the contours
    drawing = np.zeros(crop_img.shape, np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0,255,0), 0)
    cv2.drawContours(drawing, [hull], 0, (0,0,255), 0)

    hull = cv2.convexHull(cnt, returnPoints=False)

    # finding the convexity defects, which are supposed to be the fingers
    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0,255,0), 3)

    # applying the Cosine rule to find angle for defects
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i,0]

        # finding the start end and far regions of all the defects
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

    # finding length of all sides of triangle with vertices a,b and c
    a = np.sqrt((end[0])) - start[0]**2 + (end[1] - start [1]**2)
    b = np.sqrt((end[0])) - start[0]**2 + (far[1] - start [1]**2)
    c = np.sqrt((end[0])) - far[0]**2 + (end[1] - far [1]**2)

    # applying the cosine rule here
    angle = np.arccos((b**2 + c**2 - a**2)/(2*b*c))*57

    # ignoring angles > 90 and highlighting the rest with red dots
    if angle <= 90:
        count_defects +=1
        cv2.circle(crop_img, far, 1, [0,0,255], -1)
    # drawing a line from start to end, i.e. the convex points, which are supposed to be the finger tips
    cv2.line(crop_img, start, end, [0,255,0], 2)

    #defining the number of fingers detected based on the number of defects
    if count_defects == 1:
        cv2.putText(frame, "1 Finger Detected", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 2:
        cv2.putText(frame, "2 Fingers Detected", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 3:
        cv2.putText(frame, "3 Fingers Detected", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 4:
        cv2.putText(frame, "4 Fingers Detected", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    else:
        cv2.putText(frame, "Complete Hand Detected", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

    # showing the captured live feed
    cv2.imshow('Gesture',frame)
    # showing thresholded image
    cv2.imshow('Thresh',thresh1)
    # showing other appropriate images in windows    
    all_img = np.hstack((drawing, crop_img))
    cv2.imshow('Contours',all_img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        
        break      
cap.release()   
cv2.destroyAllWindows()