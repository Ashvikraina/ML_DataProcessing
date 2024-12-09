import google.generativeai as genai
apikey="AIzaSyCNAjDpbpO_TZh-_hQv4rzzW9MS60SB5yU"
genai.configure(api_key=apikey)
model=genai.GenerativeModel("gemini-1.5-flash")
chat=model.start_chat()
response=chat.send_message("What's the capital of the US")
print(response.text)
response=chat.send_message("How many people live in that city?")
print(response.text)