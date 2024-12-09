import cv2
import time
from transformers import pipeline
from datetime import datetime
import pyttsx3
engine=pyttsx3.init()
camera = cv2.VideoCapture(0)
image_to_text = pipeline("image-to-text")
def take_screenshot():
    ret, frame = camera.read()
    if ret:
        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print('hi')
        return filename
def describe_image(image_path):
    print('aloha')
    result = image_to_text(image_path)
    return result[0]["generated_text"]
while True:
    screenshot_path = take_screenshot()
    description = describe_image(screenshot_path)
    engine.say(description)
    engine.runAndWait()
    time.sleep(30)
camera.release()
cv2.destroyAllWindows()