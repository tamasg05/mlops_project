from flask import Flask, request, jsonify
import pandas as pd
import io
from MLPersist import MLPersist

CSV_DATA = "titanic.pkl"
app = Flask(__name__)
persist = MLPersist()
train_accuracy = 0
test_accuracy = 0

@app.route('/train_csv', methods=['POST'])
def train_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.csv'):
        try:
            csv_data = file.read().decode('utf-8')
            df = pd.read_csv(io.StringIO(csv_data))

            global train_accuracy, test_accuracy
            _, train_accuracy, test_accuracy = persist.train_pipeline(df.copy(), test_accuracy)

            return jsonify({'success': True, 'message':'Model trained', 'test accuracy': test_accuracy})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file'}), 400

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.csv'):
        try:
            csv_data = file.read().decode('utf-8')
            df = pd.read_csv(io.StringIO(csv_data))
            df_t = persist.preprocess_pipeline(df.copy())
            y = persist.predict(df_t)

            return jsonify({'success': True, 'predictions': y.to_dict(orient='records')})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)
