#!/usr/bin/python3



# OpenCV People Counting Program (Raspberry Pi Camera Module) 
# Written By: Harrison Pace (2018)
#
# Special Thanks to: Federico (fedemejia.com) - code is based on OpenCV Tutorials (available: https://fedemejia.com/?p=83)
# Credits: Fedemejia https://fedemejia.com/?p=83, Adrian Rosebrock (https://www.pyimagesearch.com), OpenFrameworks (OpenCV) (https://openframeworks.cc/)


#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
from person import MyPerson #based on Fedemejia Person Class
import argparse
import datetime
import imutils
import math
import numpy as np
import time
import cv2

# Establish Database Connection
import pymongo
from pymongo import MongoClient

#Create MongoDB Client
client = MongoClient()

#Set Client Connection
client = MongoClient('localhost', 27017)

#Define db
db = client.peoplecount
collection = db.peoplecount


# Define Camera Properties 
w = 640
h = 480
framerate = 5
initTime = 0.1
mx = int(w/2)
my = int(h/2)
frameArea = h*w
areaTH = frameArea/250
print ("Area Threshold", areaTH)

#Define Detection Lines 
line_up     = int(2*(h/5))
line_down   = int(3*(h/5))

#Define Detection Limits
up_limit =   int(1*(h/5))
down_limit = int(4*(h/5))

#Print Line (Y) Positions
print ("Red line y:",  str(line_down))
print ("Blue line y:", str(line_up))

#Set Line Properties
line_down_color = (255,0,0)
line_up_color = (0,0,255)
pt1 =  [0, line_down];
pt2 =  [w, line_down];
pts_L1 = np.array([pt1,pt2], np.int32)
pts_L1 = pts_L1.reshape((-1,1,2))
pt3 =  [0, line_up];
pt4 =  [w, line_up];
pts_L2 = np.array([pt3,pt4], np.int32)
pts_L2 = pts_L2.reshape((-1,1,2))

pt5 =  [0, up_limit];
pt6 =  [w, up_limit];
pts_L3 = np.array([pt5,pt6], np.int32)
pts_L3 = pts_L3.reshape((-1,1,2))
pt7 =  [0, down_limit];
pt8 =  [w, down_limit];
pts_L4 = np.array([pt7,pt8], np.int32)
pts_L4 = pts_L4.reshape((-1,1,2))

# Init Background Subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

# Initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (w, h)
camera.framerate = framerate
rawCapture = PiRGBArray(camera, size=(w, h))
 
# allow the camera to adjust
time.sleep(initTime)

# define empty first frame
firstFrame = None

#Structure Filters 
kernelOp = np.ones((3,3),np.uint8)
kernelOp2 = np.ones((5,5),np.uint8)
kernelCl = np.ones((11,11),np.uint8)

#OpenCV Variables
font = cv2.FONT_HERSHEY_SIMPLEX
persons = []
max_p_age = 5
pid = 1
cnt_up   = 0
cnt_down = 0


 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    
    #Apply Age to each Person
    for i in persons:
        i.age_one()
       
    
    # grab the raw array representing the image, then initialize the image
    image = frame.array

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    
    # debug text
    text = "Unoccupied"
    
    # Use OpenCV std name
    frame = image
    #Create 2nd Frame for testing
    frame2 = frame.copy()
    
    # Apply Background Subtractor
    fgmask = fgbg.apply(frame)
    fgmask2 = fgbg.apply(frame)
    
    #Eliminate Shadows using GreyScale Filters
    try:
        ret,imBin= cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)
        ret,imBin2 = cv2.threshold(fgmask2,200,255,cv2.THRESH_BINARY)
        #Opening (erode -> dilate)
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
        mask2 = cv2.morphologyEx(imBin2, cv2.MORPH_OPEN, kernelOp)
        #Closing (dilate -> erode)
        mask =  cv2.morphologyEx(mask , cv2.MORPH_CLOSE, kernelCl)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernelCl)
    except:
        print('FAILURE')
        print ("UP:",cnt_up)
        print ("DOWN:",cnt_down)
        break
    
    # Find Object Contours & check meets threshold     
    _, contours0, hierarchy = cv2.findContours(mask2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        area = cv2.contourArea(cnt)
        if area > areaTH:

            
            # Tracking Code 
            # Track Objects and State (defined via Person Class)
            
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            x,y,w,h = cv2.boundingRect(cnt)

            new = True
            if cy in range(up_limit,down_limit):
                for i in persons:
                    if abs(cx-i.getX()) <= w and abs(cy-i.getY()) <= h:
                        # Check if Object is within Threshold of existing Object (single entity)
                        new = False
                        i.updateCoords(cx,cy)   #Update Person Coindinates
                        #Check if Person Object Crossed Bounds
                        if i.going_UP(line_down,line_up) == True:
                            cnt_up += 1;
                            print ("ID:",i.getId(),"crossed going up at",time.strftime("%c"))
                            #Add to DB
                            post = {"id":i.getId(),"direction": "UP","date": datetime.datetime.utcnow()}
                            ins = collection.insert_one(post)
                            print(ins.inserted_id)
                        elif i.going_DOWN(line_down,line_up) == True:
                            cnt_down += 1;
                            print ("ID:",i.getId(),"crossed going down at",time.strftime("%c"))
                            #Add to DB
                            post = {"id":i.getId(),"direction": "DOWN","date": datetime.datetime.utcnow()}
                            ins = collection.insert_one(post)
                            print(ins.inserted_id)
                        break
                    #Remove Old Objects from List (Person left bounds)
                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > down_limit:
                            i.setDone()
                        elif i.getDir() == 'up' and i.getY() < up_limit:
                            i.setDone()
                    if i.timedOut():
                        #Remove from List
                        index = persons.index(i)
                        persons.pop(index)
                        del i     #Delete Object Reference
                if new == True:
                    p = MyPerson(pid,cx,cy, max_p_age)
                    persons.append(p)
                    pid += 1     
            #Drawings 
            cv2.circle(frame,(cx,cy), 5, (0,0,255), -1) #Draw Circle at Person Centre Point
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) # Draw Rectange around identified Person
    
    # Display ID for Each identified Person
    for i in persons:
        cv2.putText(frame, str(i.getId()),(i.getX(),i.getY()),font,0.3,i.getRGB(),1,cv2.LINE_AA)
        
    # Draw Overlay HUD (Lines / Text)
    str_up = 'UP: '+ str(cnt_up)
    str_down = 'DOWN: '+ str(cnt_down)
    frame = cv2.polylines(frame,[pts_L1],False,line_down_color,thickness=2)
    frame = cv2.polylines(frame,[pts_L2],False,line_up_color,thickness=2)
    frame = cv2.polylines(frame,[pts_L3],False,(255,255,255),thickness=1)
    frame = cv2.polylines(frame,[pts_L4],False,(255,255,255),thickness=1)
    cv2.putText(frame, str_up ,(10,40),font,0.5,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame, str_up ,(10,40),font,0.5,(0,0,255),1,cv2.LINE_AA)
    cv2.putText(frame, str_down ,(10,90),font,0.5,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame, str_down ,(10,90),font,0.5,(255,0,0),1,cv2.LINE_AA)

    
    # show the frames
    cv2.imshow("Frame", frame)

    # take input key (for exit method)
    key = cv2.waitKey(1) & 0xFF
    
    #Check wait input and break from loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

#Release Camera Resource        
camera.close()

#Terminate OpenCV
cv2.destroyAllWindows()