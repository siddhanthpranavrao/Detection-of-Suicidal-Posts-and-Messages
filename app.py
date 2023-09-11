import pickle
import re
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

app = Flask(__name__, template_folder='templates')

# Instantiate a lemmatizer in the global scope
lemmatizer = WordNetLemmatizer()

# Set of stopwords
stop_words = set(stopwords.words('english'))


# Function that preprocesses the given text.
def clean_text(post):
    # Splitting the sentence into words
    txt = re.sub('[^a-zA-Z]', ' ', post)  # remove special characters
    txt = txt.lower()
    txt = txt.split(' ')
    txt = [lemmatizer.lemmatize(word) for word in txt]
    # remove stop words
    txt = [word for word in txt if word not in stop_words]

    cleaned_text = ' '.join(txt)
    return cleaned_text


def predict_lstm(model, s):
    new_text = clean_text(s)
    sequence = tokenizer.texts_to_sequences([new_text])
    padded_sequence = pad_sequences(sequence, padding='post', )
    result = model.predict(padded_sequence)
    return result[0]


# Load the tokenizer
with open("models/tokenizer.pkl", "rb") as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Load your pre-trained model
lstm_model = load_model("models/lstm.h5")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_input = data.get('text', '')

        # Predict using the NLP model
        res = predict_lstm(lstm_model, user_input)
        sentiment = "Suicidal Tendencies" if res[0] > 0.6 else "No Suicidal Tendencies"

        return jsonify({'prediction': sentiment})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
