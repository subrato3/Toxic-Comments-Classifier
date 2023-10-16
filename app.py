import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the TF-IDF vectorizer and the trained model
tfidf_vectorizer = pickle.load(open("tf_idf.pkt", "rb"))
classifier_model = pickle.load(open("model.pkt", "rb"))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify_toxicity():
    comment = request.form.get('comment', '')
    # Use the TF-IDF vectorizer to transform the comment
    comment_tfidf = tfidf_vectorizer.transform([comment])

    # Use the trained model to make a prediction
    toxicity_score = classifier_model.predict_proba(comment_tfidf)[0, 1]

    # Return the toxicity score as a JSON response
    return jsonify({'toxicity_score': toxicity_score})


if __name__ == '__main__':
    app.run(debug=True)
