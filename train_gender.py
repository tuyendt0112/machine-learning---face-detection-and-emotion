
"""
GROUP 07: 
Member : 
 1/ Dang Thanh Tuyen - 20110412
 2/ Danh Truong Son - 20110394
 3/ Dang Phuoc Truong Tai - 20110396
 4/ Nguyen Trung Nguyen - 20110388
Project :  gender and emotion detection using model CNN
"""
""" 
# mo ta dataset : 
# Bộ dữ liệu UTKFace là bộ dữ liệu khuôn mặt quy mô lớn với khoảng tuổi dài (từ 0 đến 116 tuổi). 
# Bộ dữ liệu bao gồm hơn 20.000 hình ảnh khuôn mặt với các chú thích về độ tuổi, giới tính và dân tộc. 
# Các hình ảnh bao gồm sự khác biệt lớn về tư thế, nét mặt, độ chiếu sáng, độ che khuất, độ phân giải, v.v. 
# Bộ dữ liệu này có thể được sử dụng cho nhiều tác vụ khác nhau, ví dụ: nhận diện khuôn mặt, ước tính tuổi, tiến triển/hồi quy tuổi, định vị mốc, v.v. 
# có thể dowload bộ dữ liệu theo link phía dưới :  : https://susanqq.github.io/UTKFace/
# Những hình ảnh dữ liệu này thì đã được lưu trữ dưới dạng file 
# mỗi hình ảnh được lưu dưới dạng [age]_[gender]_[race]_[date&time].jpg 
# [age] : mot so nguyen tu 0 - 116 
# [gender] : 0 là male, 1 là female
# [race] : là 1 số nguyên từ 1-4, theo sắc tộc White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
# [date&time] : ngày và giờ , dưới dạng yyyymmddHHMMSSFFF, cho biết ngày mà bức ảnh được thêm vào tập UTKface

"""
#%%
# import 1 so thu vien can thiet
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPooling2D ,Dense,Dropout,BatchNormalization,Flatten,Input
from sklearn.model_selection import train_test_split
#%%
# load data from file
print("load data!")
path = "data/UTKFace"
# tiền Xử lý
images = []
age = []
gender = []
for img in os.listdir(path):
  ages = img.split("_")[0]
  genders = img.split("_")[1]
  img = cv2.imread(str(path)+"/"+str(img))
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  images.append(np.array(img))
  age.append(np.array(ages))
  gender.append(np.array(genders))
  
age = np.array(age,dtype=np.int64)
images = np.array(images)   #Forgot to scale image for my training. Please divide by 255 to scale. 
gender = np.array(gender,np.uint64)
#%%
x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(images, age, random_state=42)
x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(images, gender, random_state=42)

print("load data successful!")

#%%
print("Train model age!")
gender_model = Sequential()
gender_model.add(Conv2D(36, kernel_size=3, activation='relu', input_shape=(200,200,3)))
gender_model.add(MaxPooling2D(pool_size=3, strides=2))
gender_model.add(Conv2D(64, kernel_size=3, activation='relu'))
gender_model.add(MaxPooling2D(pool_size=3, strides=2))

gender_model.add(Conv2D(128, kernel_size=3, activation='relu'))
gender_model.add(MaxPooling2D(pool_size=3, strides=2))

gender_model.add(Conv2D(256, kernel_size=3, activation='relu'))
gender_model.add(MaxPooling2D(pool_size=3, strides=2))

gender_model.add(Conv2D(512, kernel_size=3, activation='relu'))
gender_model.add(MaxPooling2D(pool_size=3, strides=2))

gender_model.add(Flatten())
gender_model.add(Dropout(0.2))
gender_model.add(Dense(512, activation='relu'))
gender_model.add(Dense(1, activation='sigmoid', name='gender'))

gender_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_gender = gender_model.fit(x_train_gender, y_train_gender,
                        validation_data=(x_test_gender, y_test_gender), epochs=50)
print("luu model sau khi da train xong!")
gender_model.save('model_gender.h5')
print("save model successfully!")
#%%
# load lại model và kiểm tra độ chính xác
from keras.models import load_model
#Test model
my_model = load_model('model_gender.h5', compile=False)

predictions = my_model.predict(x_test_gender)
y_pred = (predictions>= 0.5).astype(int)[:,0]

from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(y_test_gender, y_pred))
# %%
