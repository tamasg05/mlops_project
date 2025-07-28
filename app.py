from flask import Flask, request, jsonify
from flask import send_file
import pandas as pd
import io
import os
from typing import Tuple

from MLPersist import MLPersist

RATIO_OF_DRIFTED_COLS = 0.3

app = Flask(__name__)
persist = MLPersist()
train_accuracy = 0
test_accuracy = 0


@app.route('/train_csv', methods=['POST'])
def train_csv():
    file = request.files.get('file')
    if file is None or file.filename == '':
        return jsonify({'error': 'No file uploaded'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file extension'}), 400

    try:
        csv_data = file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_data))
        return jsonify(train_model_from_df(df))
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
    The methods expects a csv file in the same structure as for training
    and fetches the training data from MLFlow labelled with @Staging,
    and computes the data drift based on that
    and saves it as a local file and logs it in MLFlow. 
    """
    file = request.files.get('file')
    if file is None or file.filename == '':
        return jsonify({'error': 'No file uploaded'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file extension'}), 400

    try:
        csv_data = file.read().decode('utf-8')
        current_df = pd.read_csv(io.StringIO(csv_data))

        reference_df, current_df = prepare_reference_and_current_df(persist, current_df)
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
    """Shows the data drift report in a browser"""

    if not os.path.exists(MLPersist.DRIFT_REPORT_PATH):
        return jsonify({'error': 'No report available. Run drift check first.'}), 404

    return send_file(MLPersist.DRIFT_REPORT_PATH, mimetype='text/html')

@app.route('/data_drift_summary', methods=['POST'])
def data_drift_summary():
    """
    The methods expects a csv file in the same structure as for training
    and fetches the training data from MLFlow labelled with @Staging,
    and computes the data drift based
    and sends back the summary. 
    """

    file = request.files.get('file')
    if file is None or file.filename == '':
        return jsonify({'error': 'No file uploaded'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file extension'}), 400

    try:
        csv_data = file.read().decode('utf-8')
        current_df = pd.read_csv(io.StringIO(csv_data))

        reference_df, current_df = prepare_reference_and_current_df(persist, current_df)
        summary = persist.get_data_drift_summary(reference_df, current_df)

        return jsonify(summary)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/auto_retrain_if_drifted', methods=['POST'])
def auto_retrain_if_drifted():
    """
    Automatically triggers model retraining if data drift exceeds a defined threshold.
    """

    file = request.files.get('file')
    if file is None or file.filename == '':
        return jsonify({'error': 'No file uploaded'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file extension'}), 400

    try:
        csv_data = file.read().decode('utf-8')
        current_df = pd.read_csv(io.StringIO(csv_data))

        reference_df, current_df_transformed = prepare_reference_and_current_df(persist, current_df.copy())
        summary = persist.get_data_drift_summary(reference_df, current_df_transformed)

        drift_ratio = summary.get("share_drifted", 0.0)
        if isinstance(drift_ratio, str):  # handle "n/a"
            drift_ratio = 0.0

        if drift_ratio >= RATIO_OF_DRIFTED_COLS:
            print(f"Drift ratio {drift_ratio:.2f} exceeded threshold. Retraining model.")
            result = train_model_from_df(current_df)
            result["retrained"] = True
            result["drift_ratio"] = drift_ratio
            return jsonify(result)
        else:
            return jsonify({
                "success": True,
                "message": "No retraining necessary. Drift below threshold.",
                "retrained": False,
                "drift_ratio": drift_ratio
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

#------- Helper methods-----------------------------------------------------------
def train_model_from_df(df: pd.DataFrame) -> dict:
    """Helper to train the model from a DataFrame and return a JSON-like result."""
    global train_accuracy, test_accuracy
    _, train_accuracy, test_accuracy = persist.train_pipeline(df, test_accuracy)

    return {
        'success': True,
        'message': 'Model trained',
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
    }

def prepare_reference_and_current_df(persist: MLPersist, current_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess reference and current DataFrames for drift detection."""

    reference_df = persist.load_last_training_dataframe_from_mlflow()
    reference_df = persist.cleaning_dataframe(reference_df)
    reference_df, _ = persist.transform_data(reference_df, save_encoders=False)
    reference_df = persist.select_features(reference_df, full=False)
    reference_df = persist.scale_features(reference_df)

    current_df = current_df
    current_df = persist.cleaning_dataframe(current_df)
    current_df, _ = persist.transform_data(current_df, save_encoders=False)
    current_df = persist.select_features(current_df, full=False)
    current_df = persist.scale_features(current_df)

    return reference_df, current_df



if __name__ == '__main__':
    # host="0.0.0.0" allows access from external machines (to listen on all interfaces)
    app.run(debug=True, host="0.0.0.0", port=5001)
