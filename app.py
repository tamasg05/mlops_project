from flask import Flask, request, jsonify
from flask import send_file
import pandas as pd
import io
import os

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

@app.route('/data_drift_csv', methods=['POST'])
def data_drift_csv():
    """
    Detect data drift by comparing new data to last trained data.
    Returns an Evidently HTML report.
    """
    file = request.files.get('file')
    if file is None or file.filename == '':
        return jsonify({'error': 'No file uploaded'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file extension'}), 400

    try:
        csv_data = file.read().decode('utf-8')
        current_df = pd.read_csv(io.StringIO(csv_data))

        # Align columns before drift check
        reference_df = persist.load_last_training_dataframe_from_mlflow()
        reference_df = persist.cleaning_dataframe(reference_df)
        reference_df, _ = persist.transform_data(reference_df, save_encoders=False)
        reference_df = persist.select_features(reference_df, full=False)
        reference_df = persist.scale_features(reference_df)

        current_df = persist.cleaning_dataframe(current_df)
        current_df, _ = persist.transform_data(current_df, save_encoders=False)
        current_df = persist.select_features(current_df, full=False)
        current_df = persist.scale_features(current_df)

        persist.generate_data_drift_report(reference_df, current_df)

        return jsonify({
            'success': True,
            'message': 'Data drift report generated.',
            'report_path': MLPersist.DRIFT_REPORT_PATH
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/data_drift_report', methods=['GET'])
def get_report():
    if not os.path.exists(MLPersist.DRIFT_REPORT_PATH):
        return jsonify({'error': 'No report available. Run drift check first.'}), 404

    return send_file(MLPersist.DRIFT_REPORT_PATH, mimetype='text/html')

@app.route('/data_drift_summary', methods=['POST'])
def data_drift_summary():
    """
    Return a JSON summary of data drift between the uploaded CSV and 
    the last staged training dataset.
    """
    file = request.files.get('file')
    if file is None or file.filename == '':
        return jsonify({'error': 'No file uploaded'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file extension'}), 400

    try:
        csv_data = file.read().decode('utf-8')
        current_df = pd.read_csv(io.StringIO(csv_data))

        reference_df = persist.load_last_training_dataframe_from_mlflow()

        # Align columns
        reference_df = persist.cleaning_dataframe(reference_df)
        reference_df, _ = persist.transform_data(reference_df, save_encoders=False)
        reference_df = persist.select_features(reference_df, full=False)
        reference_df = persist.scale_features(reference_df)

        current_df = persist.cleaning_dataframe(current_df)
        current_df, _ = persist.transform_data(current_df, save_encoders=False)
        current_df = persist.select_features(current_df, full=False)
        current_df = persist.scale_features(current_df)

        summary = persist.get_data_drift_summary(reference_df, current_df)
        return jsonify(summary)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # host="0.0.0.0" allows access from external machines (to listen on all interfaces)
    app.run(debug=True, host="0.0.0.0", port=5001)
