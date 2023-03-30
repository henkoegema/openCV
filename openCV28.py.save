# https://www.youtube.com/watch?v=lxFPM93Qe8o&t=277s&ab_channel=PaulMcWhorter
# Tracking Objects in OpenCV Using Contours

import cv2
print(cv2.__version__)

import numpy as np

dispW=640
dispH=480
#flip=2
#Uncomment These next Two Line for Pi Camera
#camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
#cam= cv2.VideoCapture(camSet)

#Or, if you have a WEB cam, uncomment the next line
#(If it does not work, try setting to '1' instead of '0')

cam=cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)

def nothing(x):
    pass

cv2.namedWindow('Trackbars')
cv2.moveWindow('Trackbars',1320,0)


cv2.createTrackbar('hueLow', 'Trackbars', 50,179,nothing)
cv2.createTrackbar('hueHigh', 'Trackbars', 100,179,nothing)
cv2.createTrackbar('hueLow2', 'Trackbars', 50,179,nothing)
cv2.createTrackbar('hueHigh2', 'Trackbars', 100,179,nothing)
cv2.createTrackbar('satLow', 'Trackbars',100,255,nothing)
cv2.createTrackbar('satHigh', 'Trackbars', 255,255,nothing)
cv2.createTrackbar('valLow', 'Trackbars', 100,255,nothing)
cv2.createTrackbar('valHigh', 'Trackbars', 255,255,nothing)

while True:
    ret, frame = cam.read()  # Read a frame from the camera
    #frame = cv2.imread('openCV/smarties.png')
   

    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    hueLow = cv2.getTrackbarPos('hueLow','Trackbars')
    hueHigh = cv2.getTrackbarPos('hueHigh','Trackbars')
    hueLow2 = cv2.getTrackbarPos('hueLow2','Trackbars')
    hueHigh2 = cv2.getTrackbarPos('hueHigh2','Trackbars')
    satLow = cv2.getTrackbarPos('satLow','Trackbars')
    satHigh = cv2.getTrackbarPos('satHigh','Trackbars')
    valLow = cv2.getTrackbarPos('valLow','Trackbars')
    valHigh = cv2.getTrackbarPos('valHigh','Trackbars')

    lowerBound = np.array([hueLow, satLow, valLow])
    lowerBound2 = np.array([hueLow2, satLow, valLow])
    upperBound = np.array([hueHigh, satHigh, valHigh])
    upperBound2 = np.array([hueHigh2, satHigh, valHigh])

    foreGroundMask = cv2.inRange(hsv, lowerBound, upperBound)
    foreGroundMask2 = cv2.inRange(hsv, lowerBound2, upperBound2)
    foreGroundMaskComp = cv2.add(foreGroundMask,foreGroundMask2)
    cv2.imshow('Fore Ground Mask Composite', foreGroundMaskComp)
    cv2.moveWindow('Fore Ground Mask Composite', 0, 530)

    contours, hierarchy = cv2.findContours(foreGroundMaskComp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = lambda x:cv2.contourArea(x), reverse=True)
    for contour in contours:
        area = cv2.contourArea(contour)
        (x,y,w,h) = cv2.boundingRect(contour)
        if area >= 50:
            #cv2.drawContours(frame,[contour],0,(255,0,0),3)
            cv2.rectangle(frame, (x,y), (x+w,y+h),(255,0,0),3)
    
    cv2.imshow('nanoCam', frame)  # Show the frame
    cv2.moveWindow('nanoCam',0,0)

    
    if cv2.waitKey(50) == ord('q'):  # Press 'q' to quit
        break
    
cam.release()  # Release the camera
cv2.destroyAllWindows()  # Close all windows

