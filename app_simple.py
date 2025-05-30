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
            persist.save_artifact(df, persist.MODEL_FOLDER + CSV_DATA)

            global train_accuracy, test_accuracy
            _, train_accuracy, test_accuracy = persist.train_pipeline(df.copy(), test_accuracy)

            return jsonify({'success': True, 'rows': len(df)})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
