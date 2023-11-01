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
# bộ dataset FER2013, một bộ dataset phổ biến với 35,887 grayscale ảnh khuôn mặt có kích thước 48x48 pixels.
# Bộ data gồm 7 loại: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. 
# có thể dowload bộ dữ liệu theo link phía dưới :  https://www.kaggle.com/msambare/fer2013
# Những hình ảnh dữ liệu này thì đã được lưu trữ dưới dạng file 
# gồm 2 file là tập train với tập test 
# (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral )
"""
#%%
# import 1 so thu vien can thiet 
import os
from matplotlib import pyplot as plt
import numpy as np
# import 1 so phuong thuc de load data
from keras.preprocessing.image import ImageDataGenerator
# import 1 so layer cho CNN
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D

#%
# kích thước của image 48x48
IMG_HEIGHT=48 
IMG_WIDTH = 48
batch_size=32
# load data
print("load data!")
train_data_directory='data/train/'
validation_data_directory='data/test/'
#%%
# Initialize image data generator with rescaling
# ImageDataGenerator giúp tải và gắn nhãn các tập dữ liệu hình ảnh
# ImageDataGenerator có 3 phương thức flow(), flow_from_directory() và flow_from_dataframe()
# để đọc các ảnh từ một mảng lớn numpy và thư mục chứa ảnh
# Tạo hai đối tượng cho ImageDataGenerator và cũng thay đổi tỷ lệ hình ảnh sao 
# cho giá trị pixel của chúng được chuẩn hóa từ 0 đến 1 (bằng cách chia cho 255).
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=30,shear_range=0.3,zoom_range=0.3,horizontal_flip=True,fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255)

train = train_datagen.flow_from_directory(
					train_data_directory,
					color_mode='grayscale',
					target_size=(IMG_HEIGHT, IMG_WIDTH),
					batch_size= batch_size,
					class_mode='categorical',
					shuffle=True)

validation = validation_datagen.flow_from_directory(
							validation_data_directory,
							color_mode='grayscale',
							target_size=(IMG_HEIGHT, IMG_WIDTH),
							batch_size= batch_size,
							class_mode='categorical',
							shuffle=True)
print("load data successfully!")
#%%
# kiem tra lai data 
# plot some face with labels 
# (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral )
import random
print(" plot some random face : ")
class_labels=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']
img, label = train.__next__()
i=random.randint(0, (img.shape[0])-1)
image = img[i]
labl = class_labels[label[i].argmax()]
plt.imshow(image[:,:,0], cmap='gray')
plt.title(labl)
plt.show()
#%%
print("training with CNN model !")
print("long time!")
# create model structure
# Keras Sequential Class giúp tạo thành một cụm lớp được xếp chồng tuyến tính
model = Sequential()
# Tạo Convolutionnal Layers
# Conv2D là convolution dùng để lấy feature từ ảnh với các tham số :
# filters : số filter của convolution
# kernel_size : kích thước window search trên ảnh
# activation : chọn activation như linear, softmax, relu, tanh, sigmoid
# Đầu ra của bài toán chúng ta sẽ sử dụng hàm activation để tính xác suất của ảnh đó là số gì.
# input_shape : lấy 1 ảnh với kích thước 48x48
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# Pooling Layers: Lớp hai chiều tổng hợp tối đa thực hiện các hoạt động tổng hợp tối đa cho dữ liệu không gian.
# pool_size : kích thước ma trận để lấy max hay average
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout : de tranh overfitting
model.add(Dropout(0.1))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
# Dense: Layer này cũng như một layer neural network bình thường
# units : số chiều output, như số class sau khi train ( 7 classes).
# use_bias : có sử dụng bias hay không (True or False)
# Flatten : hoạt động chuyển đổi Matrix thành mảng đơn
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
#dùng hàm softmax để chuyển sang xác suất
model.add(Dense(7, activation='softmax'))

# Hàm compile: Ở hàm này chúng ta sử dụng để training models như thuật toán train qua optimizer như Adam, SGD, RMSprop,..
# learning_rate : dạng float , tốc độc học, chọn phù hợp để hàm số hội tụ nhanh.
# categorical_crossentropy : dùng classifier nhiều class
# Tối ưu hóa là một quá trình quan trọng nhằm tối ưu các trọng số đầu vào 
# bằng cách so sánh dự đoán và hàm mất mát
model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
print( "summary : ", model.summary())

#%%\# lấy số lượng hình anh
train_path = "data/train/"
test_path = "data/test"
num_train_imgs = 0
for root, dirs, files in os.walk(train_path):
    num_train_imgs += len(files)
    
num_test_imgs = 0
for root, dirs, files in os.walk(test_path):
    num_test_imgs += len(files)

#%%
# Bao gồm data train, test đưa vào training.
# Batch_size thể hiện số lượng mẫu mà Mini-batch GD sử dụng cho mỗi lần cập nhật trọng số .
# Epoch là số lần duyệt qua hết số lượng mẫu trong tập huấn luyện.
# Một Epoch được tính là khi chúng ta đưa tất cả dữ liệu trong tập train vào mạng neural network 1 lần
# Batch size là số lượng mẫu dữ liệu trong một lần huấn luyện.

history=model.fit(train,
                steps_per_epoch=num_train_imgs//batch_size,
                epochs= 50,
                validation_data=validation,
                validation_steps=num_test_imgs//batch_size)
print("luu model sau khi da train xong!")
model.save('emotion_models.h5')
print("save model successfully!")
#%%
# kiem tra model
from keras.models import load_model
from sklearn import metrics
from sklearn.metrics import confusion_matrix
#Test  model
my_model = load_model('emotion_model.h5', compile=False)

# gọi hàm predict de du doan
test_img, test_lbl = validation.__next__()
predictions=my_model.predict(test_img)

predictions = np.argmax(predictions, axis=1)
test_labels = np.argmax(test_lbl, axis=1)
# in ra accuracy và confudion matrix
print ("Accuracy = ", metrics.accuracy_score(test_labels, predictions))
cm = confusion_matrix(test_labels, predictions)
#print(cm)
#Accuracy = 0.75
#%%
