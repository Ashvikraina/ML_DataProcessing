from flask import Flask, request, render_template, redirect, url_for
from transformers import pipeline

app = Flask(__name__)

emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", framework="pt")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_text = request.form['userInput']

        predictions = emotion_model(user_text)
        emotion = predictions[0]['label']

        return render_template('eval.html', emotion=emotion)

    return render_template('eval.html', emotion=None)

if __name__ == '__main__':
    app.run(debug=True)
