from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import string
import random
import nltk
from flask import Flask,render_template,url_for,request
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
app=Flask(__name__)

warnings.filterwarnings('ignore')

f = open('data.txt', 'r', errors='ignore')

raw = f.read()
raw = raw.lower()

sent_tokens = nltk.sent_tokenize(raw)  # converts to list of sentences
word_tokens = nltk.word_tokenize(raw)
# preprocessing
lemmer = nltk.stem.WordNetLemmatizer()

@app.route('/')
@app.route('/home')
def home():
    return render_template("index.html")

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))




@app.route("/get")
def response():
    user_response = request.args.get('msg')
    user_response = user_response.lower()
    chatbot_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words="english")
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf == 0):
        chatbot_response = chatbot_response + "I am sorry! I don't understand you"
        return chatbot_response

    else:
        chatbot_response = chatbot_response+sent_tokens[idx]
        return chatbot_response
    str(bot.response(user_response))


if __name__ == "__main__":
    app.run(debug=True)