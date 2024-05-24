import numpy as np
import cv2
import sys

from holoviews.operation import contours

TEXT_COLOR = (24,201,255)
TRACKER_COLOR = (255,128,0)
WARNING_COLOR = (24,201,255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
VIDEO_SOURCE = "videos/Pedestrians_2.mp4"

BGS_TYPES = ['GMG','MOG2','MOG','KNN','CNT']
BGS_TYPE = (BGS_TYPES[3])
def getKernel(KERNEL_TYPE):
    if KERNEL_TYPE == 'dilation':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    if KERNEL_TYPE == 'opening':
        kernel = np.ones((3,3),np.uint8)

    if KERNEL_TYPE == 'closing':
        kernel = np.ones((3, 3), np.uint8)

    return kernel

#print(getKernel('closing'))

def getfilter(img,filter):
    if filter == 'closing':
        return cv2.morphologyEx(img,cv2.MORPH_CLOSE,getKernel('closing'),iterations=2)
    if filter == 'opening':
        return cv2.morphologyEx(img,cv2.MORPH_OPEN,getKernel('opening'),iterations=2)
    if filter == 'dilation':
        return cv2.dilate(img,getKernel('dilation'),iterations=2)
    if filter == 'combine':
        closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,getKernel('closing'),iterations=2)
        opening = cv2.morphologyEx(closing,cv2.MORPH_OPEN,getKernel('opening'),iterations=2)
        dilation = cv2.dilate(opening,getKernel('dilation'),iterations=2)

        return dilation

def getBGSubtractor(BGS_TYPE):
    if BGS_TYPE == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=120,decisionThreshold=0.8)
    if BGS_TYPE == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG(history=200,nmixtures=5,backgroundRatio=0.7,noiseSigma=0)
    if BGS_TYPE == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2(history=500,detectShadows=True,varThreshold=100)

    if BGS_TYPE == 'KNN':
        return cv2.createBackgroundSubtractorKNN(history=500,dist2Threshold=400,detectShadows=True)

    if BGS_TYPE == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15,useHistory=True,maxPixelStability=15*60,isParallel = True)

    print('Detector invalido')
    sys.exit(1)

cap = cv2.VideoCapture(VIDEO_SOURCE)
bg_subtractor = getBGSubtractor(BGS_TYPE)
minArea = 400
maxArea = 3000
def main():
    while(cap.isOpened):
        ok,frame = cap.read()
        if not ok:
            print('Error')
            break

        frame = cv2.resize(frame,(0,0),fx=0.50,fy=0.50)
        bg_mask = bg_subtractor.apply(frame)
        bg_mask = getfilter(bg_mask,"combine")
        bg_mask = cv2.medianBlur(bg_mask,5)
        cv2.imshow('Frame',frame)
        cv2.imshow('Mask',bg_mask)
        (countours,hierarchy) = cv2.findContours(bg_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in countours:
            area = cv2.contourArea(cnt)
            if area >= minArea:
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.drawContours(frame,cnt,1,TRACKER_COLOR,10)
                cv2.drawContours(frame,cnt,1,(255,255,255),1)

                if area >= maxArea:
                    cv2.rectangle(frame,(x,y),(x+120,y-13),(49,49,49),-1)
                    cv2.putText(frame,'Aviso distanciamento',(x,y-2),FONT,0.4,(255,255,255),1,cv2.LINE_AA)
                    cv2.drawContours(frame,[cnt],-1,(255,255,255),1)

        res = cv2.bitwise_and(frame,frame,mask=bg_mask)
        cv2.putText(res,BGS_TYPE,(10,50),FONT,1,(255,255,255),3,cv2.LINE_AA)
        cv2.putText(res, BGS_TYPE, (10, 50), FONT, 1,TEXT_COLOR, 2, cv2.LINE_AA)

        cv2.imshow('Frame', frame)
        cv2.imshow('Mask',res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

main()

