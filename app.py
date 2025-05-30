from flask import Flask
from flask_restx import Api, Resource, reqparse
from werkzeug.datastructures import FileStorage
import pandas as pd
import io

app = Flask(__name__)

# For OpenAPI
api = Api(app, version='1.0', title='CSV Upload API',
          description='A simple API to upload and process CSV files')

ns = api.namespace('upload', description='CSV file upload operations')

upload_parser = reqparse.RequestParser()
upload_parser.add_argument('file', type=FileStorage, location='files', required=True, help='The CSV file to upload')

# OpeanAPI end point on http://127.0.0.1:5000/upload/csv/
@ns.route('/csv')
class CSVUpload(Resource):

    @ns.expect(upload_parser)
    def post(self):
        args = upload_parser.parse_args()

        file = args['file']
        if file and file.filename.endswith('.csv'):
            try:
                csv_data = file.read().decode('utf-8')
                df = pd.read_csv(io.StringIO(csv_data))
                result = {'message': 'CSV file uploaded and processed successfully',
                          'row_count': len(df),
                          'columns': df.columns.tolist()}
                return result, 200
            except Exception as e:
                return {'error': f'Error processing CSV file: {str(e)}'}, 500
        return {'error': 'Invalid file format. Only CSV files are allowed'}, 400

if __name__ == '__main__':
    app.run(debug=True)