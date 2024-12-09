from sklearn.model_selection import train_test_split
import cv2
import os
import keras
import numpy as np
# mainfolder="images"
# subfolders=os.listdir(mainfolder)
# data=[]
# labels=[]
# for e in subfolders:
#     pathtofolder=os.path.join(mainfolder,e)
#     filesinfolder=os.listdir(pathtofolder)
#     for f in filesinfolder:
#         fullpath=os.path.join(pathtofolder,f)
#         image=cv2.imread(fullpath)
#         image=cv2.resize(image,(50,50))
#         grayimage=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#         data.append(grayimage)
#         if pathtofolder=='images\pencil':
#             labels.append(0)
#         else:
#             labels.append(1)
#         # cv2.imshow("image",image)
#         # cv2.waitKey()
#         # cv2.destroyAllWindows()
# dataarray=np.array(data)
# labelsarray=np.array(labels)
# train_images,test_images,train_labels,test_labels=train_test_split(dataarray,labelsarray,test_size=0.2)
# print(train_images.shape)
# print(test_images.shape)
# print(train_labels)
# print(test_labels)
# train_images=train_images/255
# test_images=test_images/255

# model= keras.Sequential([
#     keras.layers.Flatten(input_shape=(50,50)),
#     keras.layers.Dense(128,activation='relu'),
#     keras.layers.Dense(2,activation='softmax')])

# model.compile(optimizer="adam",
# loss="sparse_categorical_crossentropy",
# metrics=["accuracy"])

# model.fit(train_images,train_labels,epochs=40)
# model.save("bookorpencil.h5")


# from keras.models import load_model
# model=load_model("bookorpencil.h5")
# import cv2
# img=cv2.imread("pencil123.jpg")
# grayimage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# roi=cv2.resize(grayimage,(50,50))
# roiarray=np.reshape(roi,(1,50,50))
# roiarray=roiarray/255
# prediction=model.predict(roiarray)
# # print(prediction[0])
# highestvalueindex=np.argmax(prediction[0])
# print(highestvalueindex)
# cv2.putText(img,str(highestvalueindex),(10,50),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),2)
# cv2.imshow("black",img)
# cv2.waitKey()
# cv2.destroyAllWindows()

import cv2
camera=cv2.VideoCapture(0)
facecascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img=cv2.imread("smily.png")
while camera.isOpened():
    ret,frame=camera.read()
    if ret:
        grayimage=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=facecascade.detectMultiScale(grayimage,1.3,5)
        for face in faces:
            b=face[1]+face[3]
            c=face[0]+face[2]
            if b is not None and c is not None:
                print('true')
                # img=cv2.resize(img,(c-face[0],b-face[1]))
                # frame[face[1]:b,face[0]:c]=img
        cv2.imshow("Video",frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
        print('false')
camera.release()
cv2.destroyAllWindows()