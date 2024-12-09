# Question 1

# import pandas as pd
# import matplotlib.pyplot as plt 
# data = pd.read_csv('world_population.csv')
# sorted_data = data[['Country/Territory', '2022 Population']].sort_values('2022 Population')
# least_10 = sorted_data.head(10)
# total_population = data['2022 Population'].sum()
# other_population = total_population - least_10['2022 Population'].sum()
# labels = list(least_10['Country/Territory']) + ['Other']
# populations = list(least_10['2022 Population']) + [other_population]
# plt.pie(populations, labels=labels, autopct='%1.1f%%')
# plt.show()
# us_data = data[data['Country/Territory'] == 'United States']
# years = ['2010 Population', '2015 Population', '2020 Population', '2022 Population']
# us_population = us_data[years].values.flatten()
# x_years = [2010, 2015, 2020, 2022]
# plt.bar(x_years, us_population)
# plt.title('Pop US')
# plt.xlabel('Year')
# plt.ylabel('Pop')
# plt.show()



# Question 2

# import pandas as pd
# import matplotlib.pyplot as plt
# data = pd.read_csv('snacks.txt')
# food_counts = data['favourite_food'].value_counts()
# plt.pie(food_counts, labels=food_counts.index, autopct='%1.1f%%')
# plt.title('Favorite Foods')
# plt.show()
# plt.hist(data['age'], bins=15, color='lightblue', edgecolor='black')
# plt.title('Ages')
# plt.xlabel('Age')
# plt.ylabel('# of times')
# plt.show()


 
# Question 3

# import cv2
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# smiley = cv2.imread('smily.png')
# camera = cv2.VideoCapture(0)
# while camera.isOpened():
#     ret, frame = camera.read()
#     if ret:
#         height, width, _ = frame.shape
#         left_half = frame[:, :width//2]
#         right_half = frame[:, width//2:]
#         mirrored_left = cv2.flip(left_half,1)
#         frame[:, width//2:] = mirrored_left
#         gray_right = cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray_right, 1.3, 5)
#         for (x, y, w, h) in faces:
#             resized_smiley = cv2.resize(smiley, (w, h))
#             frame[y:y+h, x+width//2:x+width//2+w] = resized_smiley
#         cv2.imshow('Webcam Feed', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
# camera.release()
# cv2.destroyAllWindows()



# Question 4

# import cv2
# import numpy as np
# from sklearn.model_selection import train_test_split
# import os
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras import datasets, layers, models

# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     img_resized = cv2.resize(img, (128, 128))
#     img_normalized = img_resized / 255.0
#     return img_normalized

# dataset_path = 'C:/Users/ashvi/machine learning/data processing/fruits360/fruits-360/Training'

# X = []
# y = []

# class_names = os.listdir(dataset_path)
# for idx, class_name in enumerate(class_names):
#     class_folder = os.path.join(dataset_path, class_name)
#     for img_file in os.listdir(class_folder):
#         img_path = os.path.join(class_folder, img_file)
#         X.append(preprocess_image(img_path))
#         y.append(idx)

# X = np.array(X)
# y = np.array(y)


# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(len(class_names), activation='softmax'))


# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))


# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.show()

# test_loss, test_acc = model.evaluate(X_val, y_val, verbose=2)


# model.save('fruit_classifier_model.h5')










# import os
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras import layers, models
# import matplotlib.pyplot as plt
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras import regularizers

# dataset_path = 'C:/Users/ashvi/machine learning/data processing/fruits360/fruits-360/Training'

# datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)


# train_generator = datagen.flow_from_directory(
#     dataset_path,
#     target_size=(128, 128),
#     batch_size=32,
#     class_mode='sparse',
#     subset='training'
# )

# validation_generator = datagen.flow_from_directory(
#     dataset_path,
#     target_size=(128, 128),
#     batch_size=16,
#     class_mode='sparse',
#     subset='validation'
# )

# class_names = train_generator.class_indices
# num_classes = len(class_names)


# Strategies to prevent overfitting - batch size, dropout, modify layers, add / remove layers


# final accuracy aruond .95 but doesn't work - 10 epochs - 32 batch
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dense(num_classes, activation='softmax'))

#final accuracy .25 - 5 epochs - 32 batch with dropout
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.25))
# model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.25))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.25))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(num_classes, activation='softmax'))



# final accuracy 1 - 5 epochs - worked for banana but not much else - tried something new - 32 batch
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
# model.add(layers.BatchNormalization())  # Added Batch Normalization
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())  # Added Batch Normalization
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())  # Added Batch Normalization
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.BatchNormalization())  # Added Batch Normalization
# model.add(layers.Dense(num_classes, activation='softmax'))
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# final accuracy - .89 - 3 epcohs - batch size 64 - weight regulation?
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3),kernel_regularizer=regularizers.l2(0.001)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu',kernel_regularizer=regularizers.l2(0.001)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu',kernel_regularizer=regularizers.l2(0.001)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
# model.add(layers.Dense(num_classes, activation='softmax'))





# final accuracy - .75 - 3 epcohs - batch size 16 - combined l2 + dropout
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3),kernel_regularizer=regularizers.l2(0.0001)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.5))  # Dropout after pooling
# model.add(layers.Conv2D(64, (3, 3), activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.5))
# model.add(layers.Conv2D(128, (3, 3), activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.5))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(num_classes, activation='softmax'))


# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# history = model.fit(
#     train_generator,
#     epochs=3,
#     validation_data=validation_generator
# )


# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.show()


# test_loss, test_acc = model.evaluate(validation_generator, verbose=2)
# print(f"Validation accuracy: {test_acc * 100:.2f}%")

# model.save('fruit_classifier_model.h5')













# Next part: Real-time object detection using OpenCV

import cv2
import numpy as np
import tensorflow as tf
import os
model = tf.keras.models.load_model('fruit_classifier_model.h5')

dataset_path = 'C:/Users/ashvi/machine learning/data processing/fruits360/fruits-360/Training'

class_names = os.listdir(dataset_path)

camera = cv2.VideoCapture(0)

while camera.isOpened():
    ret, frame = camera.read()
    if ret:
        x, y, w, h = 100, 100, 128, 128

        roi = frame[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi, (128, 128))
        roi_normalized = roi_resized / 255.0
        roi_reshaped = np.expand_dims(roi_normalized, axis=0)
        # cv2.imwrite("test.jpg",roi_resized)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # break
        prediction = model.predict(roi_reshaped)
        predicted_class = np.argmax(prediction)
        predicted_label = class_names[predicted_class]

        cv2.putText(frame, f"Class: {predicted_label}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow('Webcam Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()



