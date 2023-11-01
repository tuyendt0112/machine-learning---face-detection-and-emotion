
"""
GROUP 07: 
Member : 
 1/ Dang Thanh Tuyen - 20110412
 2/ Danh Truong Son - 20110394
 3/ Dang Phuoc Truong Tai - 20110396
 4/ Nguyen Trung Nguyen - 20110388
Project :  gender and emotion detection using model CNN
"""
#%%
# import 1 so thu vien can thiet
from keras.models import load_model
from time import sleep
import cv2
import numpy as np
from keras.utils.image_utils import img_to_array

#%%
print("load model!")
emotion_model = load_model('emotion_model.h5')
gender_model = load_model('model_gender.h5')
print("Loaded model from disk")
emotion_labels=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']
gender_labels =['man','woman']
print("load model successfully!")
#%%
# nhan du lieu tu real time webcam
cap=cv2.VideoCapture(0)
#cap = cv2.VideoCapture("C:/Users/Admin/Downloads/neutral.mp4")
while True:
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    ret,frame=cap.read()
    frame = cv2.resize(frame, (1280, 720))
    labels=[]
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces=face_detector.detectMultiScale(gray,1.3,5)
    #faces=face_detector.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,125,5),2)
        roi_gray_img=gray[y:y+h,x:x+w]
        roi_gray_img=cv2.resize(roi_gray_img,(48,48),interpolation=cv2.INTER_AREA)

        # lấy ảnh để dự đoán
        #scale anh
        img=roi_gray_img.astype('float')/255.0  
        img=img_to_array(img)
        img=np.expand_dims(img,axis=0)  

        preds=emotion_model.predict(img)[0] 
        label=emotion_labels[preds.argmax()] 
        label_position=(x,y - 20)
        cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(111,255,5),2)
        
        #Gender
        gen_img=frame[y:y+h,x:x+w]
        gen_img=cv2.resize(gen_img,(200,200),interpolation=cv2.INTER_AREA)
        gender_predict = gender_model.predict(np.array(gen_img).reshape(-1,200,200,3))
        gender_predict = (gender_predict>= 0.5).astype(int)[:,0]
        gender_label=gender_labels[gender_predict[0]] 

        gender_label_position=(x,y+h+30) 
        cv2.putText(frame,gender_label,gender_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
    cv2.imshow('Emotion and Gender Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
#%%
