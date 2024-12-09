# import cv2
# img=cv2.imread("th.jpg")
# print(img)
# cv2.imshow("hello",img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# import cv2
# img=cv2.imread("th.jpg")
# grayimage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# print(img)
# cv2.imshow("hello",img)
# cv2.imshow('Bye',grayimage)
# print(grayimage)
# cv2.waitKey()
# cv2.destroyAllWindows()

# import cv2
# img=cv2.imread("th.jpg")
# grayimage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# a,blackandwhiteimage=cv2.threshold(grayimage,120,255,cv2.THRESH_BINARY_INV)
# print(img)
# cv2.imshow("hello",img)
# cv2.imshow('Bye',grayimage)
# cv2.imshow('Baw',blackandwhiteimage)
# print(grayimage)
# print(blackandwhiteimage)
# cv2.waitKey()
# cv2.destroyAllWindows()

# import cv2
# img=cv2.imread("IMG_3238.jpg")
# img=cv2.resize(img,(640,600))
# grayimage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# a,blackandwhiteimage=cv2.threshold(grayimage,105,255,cv2.THRESH_BINARY_INV)
# contours,hierarchy=cv2.findContours(blackandwhiteimage.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img,contours,-1,(0,255,255),1)
# for each in contours:
#     rectObject=cv2.boundingRect(each)
#     if rectObject[3]>5 and rectObject[2]>5:
#         print(rectObject)
#         cv2.rectangle(img,(rectObject[0],rectObject[1]),(rectObject[0]+rectObject[2],rectObject[1]+rectObject[3]),(255,0,0),1)
# cv2.imshow("hello",img)
# cv2.imshow('Baw',blackandwhiteimage)
# cv2.waitKey()
# cv2.destroyAllWindows()

# import cv2
# camera=cv2.VideoCapture(0)
# while camera.isOpened():
#     ret,frame=camera.read()
#     if ret:
#         cv2.imshow("Video",frame)
#         if cv2.waitKey(1) & 0xFF==ord('q'):
#             break
# camera.release()
# cv2.destroyAllWindows()


# import cv2
# camera=cv2.VideoCapture(0)
# while camera.isOpened():
#     ret,frame=camera.read()
#     if ret:
#         grayvideo=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#         a,blackandwhitevideo=cv2.threshold(grayvideo,100,255,cv2.THRESH_BINARY_INV)
#         cv2.imshow("Video",frame)
#         cv2.imshow('gray',grayvideo)
#         cv2.imshow('baw',blackandwhitevideo)
#         if cv2.waitKey(1) & 0xFF==ord('q'):
#             break
# camera.release()
# cv2.destroyAllWindows()

# import cv2
# camera=cv2.VideoCapture(0)
# while camera.isOpened():
#     ret,frame=camera.read()
#     if ret:
#         cv2.rectangle(frame,(0,0),(320,480),(0,255,0),1)
#         b=frame[0:480,320:640]
#         b=cv2.flip(b,1)
#         frame[0:480,0:320]=b
#         cv2.imshow("Video",frame)
#         # roi=frame[0:480,0:320]
#         # cv2.imshow('roi',roi)
#         if cv2.waitKey(1) & 0xFF==ord('q'):
#             break
# camera.release()
# cv2.destroyAllWindows()

# import cv2
# img=cv2.imread("people.jpg")
# facecascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# grayimage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# faces=facecascade.detectMultiScale(grayimage,1.3,5)
# for face in faces:
#     cv2.rectangle(img,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(0,255,0),1)
# print(faces)
# cv2.imshow("hello",img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# import cv2
# camera=cv2.VideoCapture(0)
# facecascade=cv2.CascadeClassifier('haarcascade_eye.xml')
# while camera.isOpened():
#     ret,frame=camera.read()
#     if ret:
#         grayimage=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#         faces=facecascade.detectMultiScale(grayimage,1.3,5)
#         for face in faces:
#             cv2.rectangle(frame,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(0,255,0),1)
#         cv2.imshow("Video",frame)
#         if cv2.waitKey(1) & 0xFF==ord('q'):
#             break
# camera.release()
# cv2.destroyAllWindows()

# import cv2
# img=cv2.imread("man-31.jpg")
# img=cv2.resize(img,(640,480))
# facecascade=cv2.CascadeClassifier('haarcascade_smile.xml')
# grayimage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# faces=facecascade.detectMultiScale(grayimage,1.3,5)
# for face in faces:
#     cv2.rectangle(img,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(0,255,0),1)
# print(faces)
# cv2.imshow("hello",img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# import cv2
# img=cv2.imread("man-31.jpg")
# img=cv2.resize(img,(640,480))
# facecascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# smilecascade=cv2.CascadeClassifier('haarcascade_smile.xml')
# grayimage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# faces=facecascade.detectMultiScale(grayimage,1.3,5)
# for face in faces:
#     cv2.rectangle(img,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(255,255,0),1)
#     faceregion=grayimage[face[1]:face[1]+face[3],face[0]:face[0]+face[2]]
#     smiles=smilecascade.detectMultiScale(faceregion,1.3,5)
#     for smile in smiles:
#         cv2.rectangle(img,(smile[0]+face[0],smile[1]+face[1]),(smile[0]+smile[2]+face[0],smile[1]+smile[3]+face[1]),(0,255,0),1)
# cv2.imshow("hello",img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# import cv2
# a=1
# b=False
# camera=cv2.VideoCapture(0)
# while camera.isOpened():
#     ret,frame=camera.read()
#     if ret:
#         c=frame[0:240,0:320]
#         cv2.imshow("Video",frame)
#         if b:
#             cv2.imwrite("ss"+str(a)+".png",c)
#             print('screenshot taken',a)
#             a +=1
#         if cv2.waitKey(1) & 0xFF==ord('q'):
#             break
#         if cv2.waitKey(1) & 0xFF==ord('s'):
#             b=True
#         if a==50:
#             b=False
# camera.release()
# cv2.destroyAllWindows()

# a=1
# b=True
# while True:
#     print(a)
#     if b:
#         a=a+1
#         if a==10:
#             b=False
#     else:
#         a=a-1
#         if a==0:
#             b=True

# import cv2
# camera=cv2.VideoCapture(0)
# facecascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# img=cv2.imread("smily.png")
# while camera.isOpened():
#     ret,frame=camera.read()
#     if ret:
#         grayimage=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#         faces=facecascade.detectMultiScale(grayimage,1.3,5)
#         for face in faces:
#             b=face[1]+face[3]
#             c=face[0]+face[2]
#             if b is not None and c is not None:
#                 img=cv2.resize(img,(c-face[0],b-face[1]))
#                 frame[face[1]:b,face[0]:c]=img
#         cv2.imshow("Video",frame)
#         if cv2.waitKey(1) & 0xFF==ord('q'):
#             break
# camera.release()
# cv2.destroyAllWindows()

# import cv2
# camera=cv2.VideoCapture(0)
# facecascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eyecascade=cv2.CascadeClassifier('haarcascade_eye.xml')
# img=cv2.imread("thuglife.png")
# while camera.isOpened():
#     ret,frame=camera.read()
#     if ret:
#         grayimage=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#         faces=facecascade.detectMultiScale(grayimage,1.3,5)
#         for face in faces:
#             head=frame[face[1]:face[1]+face[3],face[0]:face[0]+face[2]]
#             cv2.rectangle(frame,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(255,255,0),1)
#             eyes=eyecascade.detectMultiScale(head,1.3,5)
#             if len(eyes)==2:
#                 for eye in eyes:
#                     eye1 = eyes[0]
#                     eye2 = eyes[1]
#                     if eye1[0] > eye2[0]:
#                         eye1, eye2 = eye2, eye1
#                     x1 = face[0] + eye1[0]
#                     y1 = face[1] + eye1[1]
#                     x2 = face[0] + eye2[0] + eye2[2]
#                     y2 = face[1] + eye1[1] + eye1[3]
#                     width = x2 - x1
#                     height = y2 - y1
#                     img1 = cv2.resize(img, (width, height))
#                     for i in range(height):
#                         for j in range(width):
#                             frame[y1 + i, x1 + j] = img1[i, j]
#                     cv2.rectangle(head,(eye[0],eye[1]),(eye[0]+eye[2],eye[1]+eye[3]),(0,255,0),1)
#         cv2.imshow("Video",frame)
#         if cv2.waitKey(1) & 0xFF==ord('q'):
#             break
# camera.release()
# cv2.destroyAllWindows()

# import cv2
# camera=cv2.VideoCapture(0)
# while camera.isOpened():
#     ret,frame=camera.read()
#     if ret:
#         grayimage=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#         a,blackandwhiteimage=cv2.threshold(grayimage,120,255,cv2.THRESH_BINARY_INV)
#         cv2.imshow("hello",frame)
#         cv2.imshow('Bye',grayimage)
#         cv2.imshow('Baw',blackandwhiteimage)
#         if cv2.waitKey(1) & 0xFF==ord('q'):
#             break
# camera.release()
# cv2.destroyAllWindows()
