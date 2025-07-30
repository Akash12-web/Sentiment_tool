from flask import Flask, render_template, request
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

app = Flask(__name__)

# Default dataset path
DEFAULT_DATASET = 'Tweets.csv'

def analyze_sentiments(df):
    sid = SentimentIntensityAnalyzer()
    sentiments = []

    for text in df['text']:
        score = sid.polarity_scores(str(text))
        if score['compound'] >= 0.05:
            sentiments.append('positive')
        elif score['compound'] <= -0.05:
            sentiments.append('negative')
        else:
            sentiments.append('neutral')

    df['sentiment'] = sentiments
    return df

def chatbot_reply(text):
    if "late" in text.lower() or "cancelled" in text.lower() or "lost" in text.lower():
        return "We’re sorry to hear that. Please DM us your issue and we'll assist you right away."
    return "Thank you for your feedback!"

def generate_charts(df):
    sentiment_counts = df['sentiment'].value_counts()

    # Bar Chart
    plt.figure(figsize=(6, 4))
    colors = ['green', 'red', 'gray']
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=colors)
    plt.title('Sentiment Distribution')
    plt.tight_layout()
    plt.savefig('static/charts/sentiment_distribution.png')
    plt.close()

    # Pie Chart
    plt.figure(figsize=(5, 5))
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
    plt.title('Sentiment Proportions')
    plt.tight_layout()
    plt.savefig('static/charts/sentiment_pie.png')
    plt.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    df = None
    chatbot_output = []
    uploaded_filename = None

    if request.method == 'POST':
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            df = pd.read_csv(file)
            uploaded_filename = file.filename
        else:
            df = pd.read_csv(DEFAULT_DATASET)
            uploaded_filename = DEFAULT_DATASET

        if 'text' not in df.columns:
            return render_template('index.html', error="Dataset must contain a 'text' column.")

        df = analyze_sentiments(df)
        generate_charts(df)

        chatbot_output = [
            {"text": row['text'], "sentiment": row['sentiment'], "reply": chatbot_reply(row['text'])}
            for _, row in df[df['sentiment'] == 'negative'].head(5).iterrows()
        ]

    return render_template("index.html", chatbot_output=chatbot_output, file=uploaded_filename)

if __name__ == '__main__':
    import nltk
    nltk.download('vader_lexicon')
    app.run(host='0.0.0.0', port=5000)
