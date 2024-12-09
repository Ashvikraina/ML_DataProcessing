from keras.datasets import mnist
import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np

# (train_images,train_labels),(test_images,test_labels)=mnist.load_data()
# train_images=train_images/255
# test_images=test_images/255

# plt.imshow(train_images[0])
# print(train_images.shape)
# print(train_labels[0])
# plt.show()

# cv2.imshow("hello",train_images[0])
# cv2.waitKey()
# cv2.destroyAllWindows()

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.imshow(train_images[i])
# plt.show()

# model= keras.Sequential([
#     keras.layers.Flatten(input_shape=(28,28)),
#     keras.layers.Dense(128,activation='relu'),
#     keras.layers.Dense(10,activation='softmax')])

# model.compile(optimizer="adam",
# loss="sparse_categorical_crossentropy",
# metrics=["accuracy"])

# model.fit(train_images,train_labels,epochs=5)
# model.save("Handwrittendigits.h5")

# test_loss,test_acc=model.evaluate(test_images,test_labels,verbose=2)

# predictions=model.predict(test_images)

# highest=np.argmax(predictions[0])

# model.save("handwritten_digits.h5")

# from keras.models import load_model
# model=load_model('handwritten_digits.h5')

# image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# image_blur=cv2.GaussianBlur(image_gray,(5,5),0)

# ret,im_th=cv2.threshold(im_gray,90,255,cv2.THRESH_BINARY)

# ret,im_th=cv2.threshold(im_gray,90,255,cv2.THRESH_BINARY_INV)

# ctrs,hier=cv2.findContours(im_th.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(image,ctrs,-1,(0,255,255),3)

# cv2.boundingRect(eachcontour)

# # cv2.rectangle(image,(startingpoint_x,startingpoint_y),(endingpoint_x,endingpoint_y),color,thickness)

# ROI=image[y1:y2,x1:x2]

# roi=cv2.resize(roi,(28,28))

# roi=cv2.dilate(roi,(3,3))

from keras.models import load_model
model=load_model("Handwrittendigits.h5")
import cv2
img=cv2.imread("Numbers.png")
img=cv2.resize(img,(640,480))
grayimage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
a,blackandwhiteimage=cv2.threshold(grayimage,105,255,cv2.THRESH_BINARY_INV)
# print(img)
# cv2.imshow("hello",img)
# cv2.imshow('Bye',grayimage)
# cv2.imshow('Baw',blackandwhiteimage)
# print(grayimage)
contours,hierarchy=cv2.findContours(blackandwhiteimage.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,contours,-1,(0,255,255),1)
for each in contours:
    rectObject=cv2.boundingRect(each)
    if rectObject[3]>5 and rectObject[2]>5:
        # print(rectObject)
        x1=rectObject[0]-15
        y1=rectObject[1]-15
        x2=rectObject[0]+rectObject[2]+15
        y2=rectObject[1]+rectObject[3]+15
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
            cv2.putText(img,str(highestvalueindex),(x1+25,y1+25),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),2)
cv2.imshow("hello",img)
cv2.imshow('Baw',blackandwhiteimage)
cv2.waitKey()
cv2.destroyAllWindows()