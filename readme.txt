The application is a demo how MLOps principles can be implemented. I wrote the code as a learning project. You can use it under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0) Please keep in mind that it is as is without warranty of any kind. In no event shall the author be liable for any claim, damages or other liability arising from, out of or in connection with the software or the use or other dealings in the software as put down in the license I referred to.


The major modules:
    -train_pipeline.py: it was the first step to collect the code from my jupyter notebook where I experimented with different models. This module served as the starting point but it is not used in the application.
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
        aliasing and easy fallback; however, those can also be achieved with some tricks.
    (2) feature/mlflow
        this branch contains model versioning with MLFlow. In addition, each training stores the artifacts, 
        including training metrics, training data, model parameters, label encoders, and the trained model itself.
        The code can be developed further to work with environments, e.g.: development, test, production, and then 
        the trained, tested models can be transitioned with copy_model_version() into the other environment. 
        In this context, I did not find this necessary. 
        Each model trained and saved is marked with the "Test" alias, if test_accuracy is higher then the previous version's,
        then the "Staging" alias is also attached. When you restart the application, the first version gets both as no 
        previous metric is available (it could be read from MLFlow, too, but it is not at present). If you perform 
        the training with the same data set again, then you get the same test_accuracy; consequently, only the "Test" alias is
        applied. The prediction always uses the models marked with the "Staging" alias illustrating how two or more models could 
        co-exist. The mechanism also makes an easy fallback and model changes to any version possible.

        If you use the "feature/mlflow" branch:
            -start mlflow ui like this:
            mlflow ui
                -the UI is tarted on  http://127.0.0.1:5000

            What I am using in MLFlow:
                -Experiments
                -Saving parameters, metrics and artifacts
                -Model registration and aliasing
    (3) feature/dockerized
        this branch contains a the application from feature/mlflow containerized that can be deployed in any docker compatible environment
        -You maybe have to change MLFlow's URL and port in MLPersist.py depending where you MLFlow runs.
            This points to the localhost that runs the docker container:
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
            Set the following environment variable to get to an MLFlow server running on your host outside the docker container:
            MLFLOW_TRACKING_URI=http://host.docker.internal:5000 
            
        -Build (you need docker):
            docker build -t titanic-app .

        -start mlflow ui on the machine hosting the docker container so that the REST application can start up
         if you have it elsewhere, set the MLFLOW_TRACKING_URI appropriately when starting the docker container

        -Start the application and create the docker container
             the 5001 port is mapped to the host port for the REST application
             docker run -it --rm -p 5001:5001 --name titanic-app-container --env MLFLOW_TRACKING_URI=http://host.docker.internal:5000 titanic-app
