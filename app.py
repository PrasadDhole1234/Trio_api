from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the saved model, CountVectorizer, and LabelEncoder
with open('decision_tree_model.pkl', 'rb') as model_file:
    saved_model, saved_cv, saved_label_encoder = pickle.load(model_file)

@app.route('/')
def home():
    return "Hello, World!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input text from the JSON request
        data = request.get_json()
        user_input = data['text']

        # Transform the user input using the saved CountVectorizer
        test_data_transformed = saved_cv.transform([user_input]).toarray()

        # Make predictions using the loaded model
        predicted_label = saved_model.predict(test_data_transformed)

        # Decode the predicted label back to the original class label
        predicted_class = saved_label_encoder.inverse_transform(predicted_label)

        # Return the result as JSON
        return jsonify({'input_text': user_input, 'predicted_class': predicted_class[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
