# import time
# a=time.time()
# while True:
#     if time.time()-a>5:
#         a=time.time()
#         print("hello")

# from elevenlabs import play
# from elevenlabs.client import ElevenLabs
# client=ElevenLabs()
# audio=client.generate(text="Hello my name is John")
# play(audio)

# import pyttsx3
# engine=pyttsx3.init()
# engine.say("My name is John")
# engine.runAndWait()

import google.generativeai as genai
apikey="AIzaSyCNAjDpbpO_TZh-_hQv4rzzW9MS60SB5yU"
genai.configure(api_key=apikey)
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content("Can you repeat this sentence using the words donald trump might use - a man holding a camera in his hand")
print(response.text)
