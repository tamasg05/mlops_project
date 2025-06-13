The application is a demo how MLOps principles can be implemented. I have written the code as a learning project without any guarantee.
The major parts:
    -app.py: this contains the REST-service with the train and predict pipelines
    -MLPersist.py: I put here everything that saves the model and the other artifacts; moreover, the logic for preprocessing, training and prediction
    -titanic_*.csv: The Titanic data set shared on https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv
                        It is an easy-to-understand data set about the tragedy of the Titanic


How to run the application:
-install Python 
-open a command shell, recommended GitBash
-install the packages in requirements.txt (if you install in a virtual environment, then create it first with the command:
    python -m venv .venv
    the environment is .venv here)
    -if you do not use a virtual environment, then skip this step:
    -navigate in your git project folder and type (do not forget the leading "." which tells the shell to run it in the current shell not in a new one):
        . ./.venv/Scripts/activate
        - this activates the virtual environment
-then start the REST-service:
    python app.py
-then you can send in parts or the full titanic data set with curl e.g.:
    curl -X POST -F "file=@titanic_train500.csv" http://127.0.0.1:5001/train_csv
    curl -X POST -F "file=@titanic_pred200.csv" http://127.0.0.1:5001/predict_csv 

There are 3 git branches with different functionalities:
    (1) feature/simple
        this branch is the simplest approach just for training a model and storing it on the file system, 
        the prediction pipeline will use the model saved during the training. There is no model versioning, 
        aliasing and easy fallback.
    (2) feature/mlflow
        this branch contains model versioning with MLFlow. In addition, each training stores the artifacts, 
        including training metrics, training data, model parameters, label encoders, and the trained model itself.

        If you use the "feature/mlflow" branch:
            -start mlflow ui like this:
            mlflow ui
                -the UI is tarted on  http://127.0.0.1:5000

            In MLFlow, I am using only one environment. However, in real life usually dev, staging and production environments were set up. Models can be transitioned with copy_model_version() 
            What I am using:
                -Experiments
                -Saving parameters, metrics and artifacts
                -Model registration and aliasing
    (3) feature/mlflow_docker
        this branch contains a the application from feature/mlflow containerized that can be deployed in any docker compatible environment
        
