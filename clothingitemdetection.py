from keras.datasets import fashion_mnist
import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np

# (train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()
# train_images=train_images/255
# test_images=test_images/255

# model= keras.Sequential([
#     keras.layers.Flatten(input_shape=(28,28)),
#     keras.layers.Dense(128,activation='relu'),
#     keras.layers.Dense(10,activation='softmax')])

# model.compile(optimizer="adam",
# loss="sparse_categorical_crossentropy",
# metrics=["accuracy"])

# model.fit(train_images,train_labels,epochs=10)
# model.save("Clothingitems.h5")

from keras.models import load_model
model=load_model("Clothingitems.h5")
import cv2
img=cv2.imread("Tshirt.jpg")
img=cv2.resize(img,(640,480))
grayimage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
a,blackandwhiteimage=cv2.threshold(grayimage,150,255,cv2.THRESH_BINARY_INV)
contours,hierarchy=cv2.findContours(blackandwhiteimage.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,contours,-1,(0,255,255),1)
for each in contours:
    rectObject=cv2.boundingRect(each)
    if rectObject[3]>5 and rectObject[2]>5:
        # print(rectObject)
        x1=rectObject[0]
        y1=rectObject[1]
        x2=rectObject[0]+rectObject[2]
        y2=rectObject[1]+rectObject[3]
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),1)
        roi=blackandwhiteimage[y1:y2,x1:x2] # dataset has black and white 
        if roi.any():
            roi=cv2.resize(roi,(28,28))
            roiarray=np.reshape(roi,(1,28,28))
            roiarray=roiarray/255
            prediction=model.predict(roiarray)
            # print(prediction[0])
            highestvalueindex=np.argmax(prediction[0])
            print(highestvalueindex)
            cv2.putText(img,str(highestvalueindex),(x1,y1+100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),2)
cv2.imshow("hello",img)
cv2.imshow('Baw',blackandwhiteimage)
cv2.waitKey()
cv2.destroyAllWindows()