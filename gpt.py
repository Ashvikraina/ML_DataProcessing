from transformers import pipeline
# text_generation=pipeline("text-generation")
# prompt="I am going to"
# result=text_generation(prompt,max_length=20)
# print(result[0]["generated_text"])

# sentiment_analysis= pipeline("sentiment-analysis")
# prompt=input()
# result=sentiment_analysis(prompt)
# print(result)

# zero_shot_classification=pipeline("zero-shot-classification")
# prompt="I was very prepared for my boxing game but at some point I felt it was going to be a tough fight"
# labels=["happy","sad","scared","angry","surprised","disgust"]
# result=zero_shot_classification(prompt,labels)
# print(result)

# zero_shot_classification=pipeline("zero-shot-classification")
# prompt="The basketball game was called off because of snow"
# labels=["weather","sports"]
# result=zero_shot_classification(prompt,labels)
# print(result)

image_to_text=pipeline("image-to-text")
url="https://letsenhance.io/static/8f5e523ee6b2479e26ecc91b9c25261e/1015f/MainAfter.jpg"
result=image_to_text(url)
print(result)