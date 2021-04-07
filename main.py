import numpy as np
import cv2
import datetime
from tensorflow.keras.models import load_model
from PIL import Image

mymodel = load_model('model.h5')
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while cap.isOpened():
    _, img=cap.read()
    face = face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4)
    for(x,y,w,h) in face:
        face_img = img[y:y+h, x:x+w]
        cv2.imwrite('temp.jpg',face_img)
        test_image = Image.open('temp.jpg')
        test_image = np.array(test_image.resize((150, 150)))
        test_image = np.array([test_image])
        pred=mymodel.predict_classes(test_image)[0][0]
        
        if pred==1:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(img,'MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(img,'MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
            datet=str(datetime.datetime.now())
            cv2.putText(img,datet,(400, 400),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            cv2.imshow('img',img)
        
        if cv2.waitKey(1)==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()