from flask import Flask, request, jsonify
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import AdamW
import tensorflow as tf
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from tensorflow.keras.layers import Layer
from flask_cors import CORS

# Download NLTK resources
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Custom Attention Layer
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        e = tf.tanh(tf.matmul(inputs, self.W) + self.b)
        alpha = tf.nn.softmax(e, axis=1)
        context = inputs * alpha
        context = tf.reduce_sum(context, axis=1)
        return context

# Load model and components
try:
    with open("sentiment_model_bundle.pkl", "rb") as f:
        model_bundle = pickle.load(f)
    model_json = model_bundle['model_json']
    weights_path = model_bundle['weights_path']
    tokenizer = model_bundle['tokenizer']
    label_encoder = model_bundle['label_encoder']
    model = model_from_json(model_json, custom_objects={'Attention': Attention})
    model.load_weights(weights_path)
    model.compile(loss='categorical_crossentropy', optimizer=AdamW(learning_rate=0.0003, weight_decay=0.01), metrics=['accuracy'])
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Text preprocessing function
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        review = data.get('review', '')
        if not review:
            return jsonify({'error': 'No review provided'}), 400

        # Preprocess and predict
        processed_review = preprocess_text(review)
        sequence = tokenizer.texts_to_sequences([processed_review])
        padded = pad_sequences(sequence, maxlen=250, padding='post', truncating='post')
        prediction = model.predict(padded, verbose=0)
        sentiment = label_encoder.inverse_transform([np.argmax(prediction[0])])[0]
        probabilities = prediction[0].tolist()

        return jsonify({
            'sentiment': sentiment,
            'probabilities': {
                'negative': probabilities[0],
                'neutral': probabilities[1],
                'positive': probabilities[2]
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)