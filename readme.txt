The application is a demo how MLOps principles can be implemented. I have written the code as a learning project without any guarantee.
The major parts:
    -app_simple.py: this contains the REST-service with the train and predict pipelines
    -MLPersist.py: I put here everything that saves the model and the other artifacts; moreover, the logic for preprocessing, training and prediction
    -titanic_*.csv: The Titanic dat set shared on https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv
                        It is an easy-to-understand data set about the tragedy of the Titanic


How to run the application:
-open a command shell, recommended GitBash
-navigate in your git project folder and type (do not forget the leading "." which tells the shell to run it in the current shell not in a new one):
    . ./.venv/Scripts/activate
    - this starts the virtual environment
-then start the REST-service:
    python app_simple.py
-then you can send in parts or the full titanic data set with curl e.g.:
    curl -X POST -F "file=@titanic_train500.csv" http://127.0.0.1:5001/train_csv
    curl -X POST -F "file=@titanic_pred200.csv" http://127.0.0.1:5001/predict_csv 

If you use the "feature/mlflow" branch:
    -start mlflow ui like this:
    mlflow ui
        -the UI is tarted on  http://127.0.0.1:5000

    In MLFlow, I am using only one environment. However, in real life usually dev, staging and production environments were set up. Models can be transitioned with copy_model_version() 
    What I am using:
        -Experiments
        -Saving parameters, metrics and artifacts
        -Model registration and aliasing

        
