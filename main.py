import cv2 as cv
import numpy as np
import math

blurValue = 19  # GaussianBlur parameter
threshold = 60  #  BINARY threshold
inAngleMin = 200
inAngleMax = 300
angleMin = 180
angleMax = 359
lengthMin = 10
lengthMax = 80

hsv = None
image_mask = None
pixel = (0,0,0) #RANDOM DEFAULT VALUE

ftypes = [
    ('JPG', '*.jpg;*.JPG;*.JPEG'), 
    ('PNG', '*.png;*.PNG'),
    ('GIF', '*.gif;*.GIF'),
]

cap = cv.VideoCapture(0)
width = 0
height = 0

def rectangles(frame, cnt, hull):
  rect = cv.minAreaRect(cnt)
  box = cv.boxPoints(rect)
  box = np.int0(box)
  cv.drawContours(frame,[box],0,(0,0,255),2)
  x,y,w,h  = cv.boundingRect(hull);
  cv.rectangle(frame, (x,y),(x+w,y+h), (0, 0, 255));


def definePoints(frame, cnt, hullIndexes, center):
  convexityDefects  = cv.convexityDefects(cnt, hullIndexes);
  count = 0                                     # count is keeping track of number of defect points
  
  if type(convexityDefects) != type(None):
    lastFar = None
    count = 0
    for i in range(convexityDefects.shape[0]):  # count is keeping track of number of defect points
        s,e,f,d = convexityDefects[i,0]                 
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        if lastFar is None:
          lastFar = far
        dist = math.hypot(far[0]-start[0], far[1] - start[1])
        if dist > 20 and d > 20:     
          cv.circle(frame, far, 5, (0, 0, 255), 2)
          lastFar = far
          count = count+1
        
          cv.line(frame, start, center, (0, 0, 255), 3 )
    #print(count)
        #dist = math.hypot(x2-x1, y2-y1)

def write(img, text):
  font = cv.FONT_HERSHEY_SIMPLEX
  cv.putText(img,text,(10,500), font, 4,(255,255,255),2,cv.LINE_AA)

def sendToArduino(center, frame):
  x, y = center
  left, right, up, down = getControlPoints()
  if x < left:
    print("left")
  
  if x > right:
    print("right")

  if y > down:
    print("down")
  
  if y < up:
    print("up")
  
  cv.line(frame,(left,0),(left,height),(255,0,0),5)
  cv.line(frame,(right,0),(right,height),(255,0,0),5)
  cv.line(frame,(0,up),(width,up),(255,0,0),5)
  cv.line(frame,(0,down),(width,down),(255,0,0),5)

def drawSegments(frame):
  left, right, up, down = getControlPoints()
  cv.line(frame,(left,0),(left,height),(255,0,0),5)
  cv.line(frame,(right,0),(right,height),(255,0,0),5)
  cv.line(frame,(0,up),(width,up),(255,0,0),5)
  cv.line(frame,(0,down),(width,down),(255,0,0),5)

def getControlPoints():
  left = int(( width / 2 ) - 50)
  right = int(( width / 2 ) + 50)
  up = int(( height / 2) - 50)
  down = int(( height / 2) + 50)
  return left, right, up, down

def pick_color(event,x,y,flags,param):
  if event == cv.EVENT_LBUTTONDOWN:
    pixel = hsv[y,x]

    #HUE, SATURATION, AND VALUE (BRIGHTNESS) RANGES. TOLERANCE COULD BE ADJUSTED.
    upper =  np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
    lower =  np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])
    #A MONOCHROME MASK FOR GETTING A BETTER VISION OVER THE COLORS 
    image_mask = cv.inRange(hsv,lower,upper)
    cv.imshow("Mask",image_mask)

def adjust_gamma(image, gamma=1.0):
   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv.LUT(image, table)


def main():
  global hsv, pixel, image_mask, width, height

  while(1):
      # Take each frame
      _, frame = cap.read()
      
      width, height = frame.shape[1::-1]
      write(frame, '{0} x {1}'.format(width,height))
      # Convert BGR to HSV
      hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
      #frame = hsv

      gamma = 0.5                                   # change the value here to get different result
      frame = adjust_gamma(frame, gamma=gamma)

      
      # define range of blue color in HSV
      lower_blue = np.array([80,50,50])
      upper_blue = np.array([120,255,255])
      # Threshold the HSV image to get only blue colors
      image_mask = cv.inRange(hsv, lower_blue, upper_blue)
      # Bitwise-AND mask and original image
      blur = cv.GaussianBlur(image_mask, (blurValue, blurValue), 0)

      res = cv.bitwise_and(frame,frame, mask= image_mask)
      ret, thresh = cv.threshold(blur, threshold, 255, cv.THRESH_BINARY)
      
      im2,contours,hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
      
      largestContour = 0
      for c in range(len(contours)):
        #print(cv.contourArea(contours[c]))
        if cv.contourArea(contours[c]) > cv.contourArea(contours[largestContour]):
          largestContour = c;

      cv.drawContours(frame, contours, largestContour, (0, 0, 255), 1);
      length = len(contours)
      if length > 0:
        cnt = contours[largestContour]

        # Calcula y dibuja la envolvente convexa https://es.wikipedia.org/wiki/Envolvente_convexa
        hull = cv.convexHull(cnt, clockwise=False)  
        cv.drawContours(frame, [hull], 0, (0, 255, 0), 5);

        if len(hull) > 0:
          hullIndexes = cv.convexHull(cnt, clockwise=False, returnPoints = False);

          rectangles(frame, cnt, hull)

          #Calcular centro
          x,y,w,h  = cv.boundingRect(hull);
          center_y = int(y+(h/2))
          center_x = int(x+(w/2))
          center = (center_x,center_y)
          # Dibujar circulo 
          cv.circle(frame, (center_x, center_y), int(w/2) ,(255, 255, 0), 5);
          print(center)
          definePoints(frame, cnt, hullIndexes, center)
          sendToArduino(center, frame)
      
      drawSegments(frame)
      cv.imshow('blur', blur)
      cv.imshow('mask', image_mask)
      cv.imshow('ori', thresh)
      cv.imshow('ret', ret)
      cv.imshow('img', frame)
      
      #cv.setMouseCallback("img", pick_color)

      # get the coutours
      #thresh1 = copy.deepcopy(thresh)
      #_,contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      
      k = cv.waitKey(5) & 0xFF
      if k == 27:
          break

if __name__=='__main__':
    main()         
    cv.destroyAllWindows()