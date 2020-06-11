
from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time

goodLower = (9,81,102)  
goodUpper = (100, 255, 255)

burntLower = (0,0,0)
burntUpper = (250,243,68)

lower_chocolat=(0,0,0) 
upper_chocolat=(16,255,121)

def Grab_HSV_Space_Contours(hsv_space,colorLower,colorUpper):
	mask = cv2.inRange(hsv_space, colorLower, colorUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	if(colorUpper==upper_chocolat):
		cnts,_ = cv2.findContours(mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	else:
		cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	return cnts

def Color_Test(hsv_space):
	cnts=Grab_HSV_Space_Contours(hsv_space,burntLower,burntUpper)
	cnts = imutils.grab_contours(cnts)
	if len(cnts) > 0 :
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		if radius > 130:
			return (1,(x, y),radius)
		else :
			return (0,(0, 0),0)
	return (0,(0, 0),0)

def Dimension_Test(hsv_space):
	cnts=Grab_HSV_Space_Contours(hsv_space,goodLower,goodUpper)
	cnts = imutils.grab_contours(cnts)
	if len(cnts) > 0 :
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		if (radius > 120) and (radius < 140) :
			return (0,1,(x, y),radius)
		else :
			return (1,0,(x, y),radius)
	return (0,0,(0, 0),0)

def Choclat_Test(hsv_space):
	contours=Grab_HSV_Space_Contours(hsv_space,lower_chocolat,upper_chocolat)
	nbre_contours= 0
	for contour in contours:
		area=cv2.contourArea(contour)
		if area>500: 
			nbre_contours+=1			
	if(nbre_contours == 3):
		return 0
	else :
		return 1

def Broken_Test(img): 
	hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	mask=cv2.inRange(hsv,goodLower,goodUpper)
	contours,hierarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	drawing = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
	for i in range(len(contours)):
		color_old_contours = (0, 255, 0) 
		cv2.drawContours(drawing, contours, i, color_old_contours, 1, 8, hierarchy)
	canny = cv2.Canny(drawing, 50, 200, None, 3) 
	linesP = cv2.HoughLinesP(canny, 1, np.pi /180, 60, None, 50, 12)
        
	if linesP is not None:
		if len(linesP)>1 :  
			return 1
		else :
			return 0
	else:
		return 0

vs = VideoStream(src=1).start()
time.sleep(2.0)

while True:
	frame = vs.read()
	
	frame = imutils.resize(frame, width=800)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	msg='Accepted'
	defected=0
	accepted=0
	center=(0,0)

	(defected,center,radius)=Color_Test(hsv)
	if(defected):
		msg='Rejected: Burnt biscuit'
	else:
		(defected,accepted,center,radius)=Dimension_Test(hsv)
		if(defected):
			msg='Rejected: Wrong dimension'
		else:
			defected=Choclat_Test(hsv)
			if(defected):
				msg='Rejected: Missing Choclat'
			else:
				defected=Broken_Test(frame)
				if(defected):
					msg='Rejected: Broken biscuit'
					defected=1

	(x,y)=center
	if (defected):
		cv2.circle(frame, (int(x), int(y)), int(radius),(0,0,255), 2)
		cv2.putText(frame,msg,(int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
	elif (accepted):
		cv2.circle(frame, (int(x), int(y)), int(radius),(0,255,0), 2)
		cv2.putText(frame,msg,(int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

	time.sleep(0.03)

vs.release()
cv2.destroyAllWindows()
