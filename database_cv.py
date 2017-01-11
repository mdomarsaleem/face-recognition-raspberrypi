import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import scipy.misc
import cPickle
import os
import time
os.chdir("//home//pi/Desktop//Image_db/")


import warnings
warnings.filterwarnings('error', category=DeprecationWarning)

def rgb2gray(rgb):
    return np.dot(rgb[:,:,:], [0.299, 0.587, 0.114])

def standard(X):
    return (X - X.mean())/X.max()

def Pre_Process(face):
    from skimage.transform import resize
    X = standard(resize(face,(96,96))).reshape(-1,1,96,96)
    X_normal = X.reshape(-1,9216)
    return X,X_normal

# load it again
with open('/home/pi/Desktop/files/linear_model.pkl', 'rb') as fid:
    Net = cPickle.load(fid)
    
map = np.load('/home/pi/Desktop/files/map.npy')
#print map

#face_cascade = cv2.CascadeClassifier('/home/pi/Downloads/opencv-2.4.13/data/haarcascades_GPU/haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier('/home/pi/Downloads/opencv-2.4.13/data/haarcascades/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('/home/pi/Downloads/opencv-2.4.13/data/lbpcascades/lbpcascade_frontalface.xml')
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (1000, 750)
camera.framerate = 15
camera.zoom = (0,0,0.75,0.75)
rawCapture = PiRGBArray(camera, size=(1000, 750))
cv2.namedWindow('Video',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video',640,480)
i = 0


for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
   frame = frame.array
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   start_time = time.time()
   faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(90, 90)
    )
   #print("--- %s seconds ---" % (time.time() - start_time))

    # Draw a rectangle around the faces
   if len(faces)>0:    
        for (x, y, w, h) in faces:
            i +=1
            fac = np.array(frame)[y:(y+h),x:(x+h),:]
            fac_gray = np.array(gray)[y:(y+h),x:(x+h)]
            X,X_normal = Pre_Process(fac_gray)
            Probability = Net.predict_proba(X.reshape(-1,9216))
            prob = np.amax(Probability)
            #print Probability
            index = np.argmax(Probability)
            #print index
            cv2.rectangle(frame, (x, y), (x+h, y+w), (0, 255, 0), 2)
            #cv2.putText(frame,'omar',(x,y+h), cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255), 2,8)
  	    #cv2.putText(frame,str(map[index])+' '+str(round(prob*100,2) )+'%',(x,y), cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255), 1,2)
  	    print("--- %s seconds ---" % (time.time() - start_time))
            scipy.misc.toimage(cv2.cvtColor(fac, cv2.COLOR_RGB2BGR)).save(time.strftime('%Y-%m-%d')+'_'+str(i) +'.jpg')

    # Display the resulting frame
   cv2.imshow('Video', frame)
   #time.sleep(0.1)
# clear the stream in preparation for the next frame
   rawCapture.truncate(0)
   if cv2.waitKey(1) & 0xFF == ord('q'):
            break
   if time.localtime(time.time()).tm_hour == 20:
            break
	    #os.system("shutdown now -h")        


# When everything is done, release the capture
cv2.destroyAllWindows()
