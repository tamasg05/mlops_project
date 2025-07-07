from flask import Flask, request, jsonify
import pandas as pd
import io

from MLPersist import MLPersist

app = Flask(__name__)
persist = MLPersist()
train_accuracy = 0
test_accuracy = 0


@app.route('/train_csv', methods=['POST'])
def train_csv():
    """
    Train the KNN model using uploaded CSV data.
    Expects a CSV file with 'survived' column.
    """
    file = request.files.get('file')
    if file is None or file.filename == '':
        return jsonify({'error': 'No file uploaded'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file extension'}), 400

    try:
        csv_data = file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_data))

        global train_accuracy, test_accuracy
        _, train_accuracy, test_accuracy = persist.train_pipeline(
            df, test_accuracy
        )

        return jsonify({
            'success': True,
            'message': 'Model trained',
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    """
    Make predictions using uploaded CSV data.
    Expects a CSV file *without* the 'survived' column.
    """
    file = request.files.get('file')
    if file is None or file.filename == '':
        return jsonify({'error': 'No file uploaded'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file extension'}), 400

    try:
        csv_data = file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_data))

        df = persist.preprocess_pipeline(df)
        predictions = persist.predict(df)

        if predictions is None:
            return jsonify({
                'error': 'The model is not available. Train the model first.'
            }), 400

        return jsonify({
            'success': True,
            'predictions': predictions.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # host="0.0.0.0" allows access from external machines (to listen on all interfaces)
    app.run(debug=True, host="0.0.0.0", port=5001)
