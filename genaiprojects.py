import cv2
import time
from transformers import pipeline
import pyttsx3
import google.generativeai as genai
apikey="AIzaSyCNAjDpbpO_TZh-_hQv4rzzW9MS60SB5yU"
genai.configure(api_key=apikey)
model = genai.GenerativeModel('gemini-1.5-flash')
engine=pyttsx3.init()
a=time.time()
image_to_text = pipeline("image-to-text")
def take_screenshot():
    ret, frame = camera.read()
    if ret:
        filename = "ss.png"
        return filename
def describe_image(image_path):
    result = image_to_text(image_path)
    return result[0]["generated_text"]
camera=cv2.VideoCapture(0)
while camera.isOpened():
    ret,frame=camera.read()
    if ret:
        cv2.imshow("Video",frame)
        if time.time()-a>30:
            a=time.time()
            cv2.imwrite("ss.png",frame)
            print('screenshot taken')
            screenshot_path = take_screenshot()
            description = describe_image(screenshot_path)
            print(description)
            response = model.generate_content("Please rephrase this sentence using words Donald Trump might use:"+description)
            print(response.text)
            engine.say(response.text)
            engine.runAndWait()
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
camera.release()
cv2.destroyAllWindows()