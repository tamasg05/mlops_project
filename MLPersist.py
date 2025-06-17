import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.entities.model_registry import ModelVersion
from mlflow.data.pandas_dataset import PandasDataset



from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

import os
from typing import Optional, Tuple, Dict
from datetime import datetime
from pathlib import Path

class MLPersist:
    MODEL_FOLDER = "artifacts/"
    LABEL_ENCODER = "label_encoder_dict.pkl"
    LABEL_ENCODER_FOLDER = "label_encoder_dictionary"
    KNN_MODEL = "knn_classifier.pkl"

    MLFLOW_NAME = "Titanic_KNN_Model"

    MLFLOW_ALIAS_STAGING = "Staging"
    MLFLOW_ALIAS_PRODUCTION = "Production"
    MLFLOW_ALIAS_TEST = "Test"


    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Titanic KNN classification tests")

    def __init__(self):
        if os.path.exists(MLPersist.MODEL_FOLDER):
            print(MLPersist.MODEL_FOLDER + " exists.")
        else:
            Path(MLPersist.MODEL_FOLDER).mkdir(parents=True, exist_ok=True)

    
    def save_artifact(self, df: pd.DataFrame, neighbours: int, train_accuracy: float, test_accuracy: float, 
                      X_train: pd.DataFrame, y_train: pd.DataFrame, knn: KNeighborsClassifier, save_threshold: float, label_encoder_path: str) -> None:
        # saving the training data, so that we could log the data set as an artifact
        train_data_path = os.path.join(self.MODEL_FOLDER, "training_data.csv")
        df.to_csv(train_data_path, index=False) 
        print(f"Temporary training data saved to: {train_data_path}")
        
        mlflow_client = mlflow.tracking.MlflowClient()            

        # check whether the model exists in MLFlow to avoid error when creating a new version of the model
        try:
            mlflow_client.get_registered_model(name=MLPersist.MLFLOW_NAME)
        except mlflow.exceptions.RestException:
            mlflow_client.create_registered_model(name=MLPersist.MLFLOW_NAME)

        try:
            # Setting a unique name for the run
            rname = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with mlflow.start_run(run_name=rname) as run:
                mlflow.log_param("Neighbors for KNN", neighbours)
                mlflow.log_metric("train_accuracy", train_accuracy)
                mlflow.log_metric("test_accuracy", test_accuracy)
                
                # logging training data set to appear on the first page in MLFlow under the experiment
                # it only shows the data types and dimensions etc. but not the data themselves
                mlflow.log_input(mlflow.data.from_pandas(df), context="training")

                # also logging here to make the data easily downloadable with the other artifacts
                mlflow.log_artifact(train_data_path, artifact_path="training_data")

                # making the label encoder dictionary available as a pickle file
                mlflow.log_artifact(label_encoder_path, artifact_path=MLPersist.LABEL_ENCODER_FOLDER)

                # setting the input data shapes
                signature = mlflow.models.infer_signature(X_train, y_train)


                mlflow.sklearn.log_model(sk_model=knn, 
                                        artifact_path="knn_model",
                                        signature=signature)
                model_uri = f"runs:/{run.info.run_id}/knn_model"
                
                model_version = mlflow_client.create_model_version(MLPersist.MLFLOW_NAME, 
                                                              model_uri, 
                                                              run.info.run_id)
                print(f"model_version.version={model_version.version}")

                if test_accuracy > save_threshold:                                                    
                    # Set registered model alias for staging and test
                    mlflow_client.set_registered_model_alias(MLPersist.MLFLOW_NAME, 
                                                             MLPersist.MLFLOW_ALIAS_TEST, 
                                                             model_version.version)

                    mlflow_client.set_registered_model_alias(name=MLPersist.MLFLOW_NAME,
                                                             alias=MLPersist.MLFLOW_ALIAS_STAGING,
                                                             version=model_version.version)
                    print(f"MLflow: Model Name: {MLPersist.MLFLOW_NAME}, Model Version: {model_version.version} registered and aliased: '{MLPersist.MLFLOW_ALIAS_STAGING}'.")               
                    print("MLflow: Trained model saved.")
                else:
                    # Set registered model alias for test only
                    mlflow_client.set_registered_model_alias(MLPersist.MLFLOW_NAME, 
                                                             MLPersist.MLFLOW_ALIAS_TEST, 
                                                             model_version.version)

                    print(f"MLflow: Model Name: {MLPersist.MLFLOW_NAME}, Model Version: {model_version.version} registered and aliased: '{MLPersist.MLFLOW_ALIAS_TEST}'.")               
                    print("MLflow: Trained model saved.")
                    

            print(f"Train accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

        except Exception as e:
            print(f"Failed saving the model with the exception: {str(e)}")
            raise # to see stack trace
            
    def save_artifact_to_disk(self, artifact: any, file_path: str) -> None:
        with open(file_path, 'wb') as file:
            pickle.dump(artifact, file)

    def load_artifact_from_disk(self, file_path: str) -> any:
        with open(file_path, 'rb') as file:
            artifact = pickle.load(file)
        return artifact
    
    def load_label_encoders(self, alias:str) -> Dict[str, LabelEncoder]:
        try:
            name = MLPersist.MLFLOW_NAME

            # Get the model version by its alias to find the associated run_id
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_model_version_by_alias(name=name, alias=alias)

            # Extract the run_id from the model version's source URI
            source_uri_parts = model_version.source.split('/')
            if len(source_uri_parts) >= 2 and source_uri_parts[0] == 'runs:':
                run_id = source_uri_parts[1]
            else:
                raise ValueError(f"Could not extract run_id from model source URI: {model_version.source}")

            full_artifact_uri = f"runs:/{run_id}/{MLPersist.LABEL_ENCODER_FOLDER}/{MLPersist.LABEL_ENCODER}"
            print(f"Loading label encoders from MLflow URI: {full_artifact_uri}")

            # Load the artifact (it downloads it locally)
            local_download_path = mlflow.artifacts.download_artifacts(artifact_uri=full_artifact_uri)

            # Deserialize the dictionary using pickle
            with open(local_download_path, 'rb') as f:
                label_encoders = pickle.load(f)
            print("Label encoders loaded successfully.")
            return label_encoders

        except Exception as e:
            print(f"Failed to load label encoders from MLflow: {e}")
            raise # Re-raise the exception to indicate failure


    def cleaning_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # converting deck to category type
        df['deck'] = df['deck'].astype('category')

        # replacing NAs in column Deck
        df['deck'] = df["deck"].cat.add_categories("None").fillna("None")

        # replacing NAs with the median age in column Age
        df['age'] = df["age"].fillna(df['age'].median())

        # dropping the remaining rows with NAs
        df = df.dropna()

        return df

    def transform_data(self, df: pd.DataFrame, alias=MLFLOW_ALIAS_STAGING, saveEncoders=True) -> Tuple[pd.DataFrame, str]:
        # encoding the "Deck" column
        col = 'deck'
        transformed_as_df = pd.get_dummies(df[col])
        coded_column_names = [col + "_" + column for column in transformed_as_df.columns]
        transformed_as_df.columns = coded_column_names

        df = pd.concat([df, transformed_as_df], axis=1)

        local_label_encoder_path = MLPersist.MODEL_FOLDER + MLPersist.LABEL_ENCODER
        # encoding adult_male and alone columns    
        if saveEncoders: 
            label_encoders = {}
            for col in ["adult_male", "alone"]:
                label_encoders[col] = LabelEncoder()
                df[col] = label_encoders[col].fit_transform(df[col])
            
            self.save_artifact_to_disk(label_encoders, local_label_encoder_path)
        else:
            label_encoders = self.load_label_encoders(MLPersist.MLFLOW_ALIAS_STAGING)
            for col in ["adult_male", "alone"]:
                df[col] = label_encoders[col].transform(df[col])

        return (df, local_label_encoder_path)

    def select_features(self, df: pd.DataFrame, full=True) -> pd.DataFrame:
        if full:
            features = ['pclass', 'adult_male', 'alone', 'fare', 'deck_A', 'deck_B', 'deck_C', 'deck_D', 'deck_E', 'deck_F', 'deck_G', 'deck_None', 'survived']
        else:
            features = ['pclass', 'adult_male', 'alone', 'fare', 'deck_A', 'deck_B', 'deck_C', 'deck_D', 'deck_E', 'deck_F', 'deck_G', 'deck_None']
            
        df = df[features]
        return df

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Normalization
        min_max_scaler = MinMaxScaler(feature_range=(0, 10))
        col = 'fare'
        df[col] = min_max_scaler.fit_transform(df[[col]])
        return df
    
    def train_pipeline(self, df: pd.DataFrame, save_threshold=0.0) -> Optional[Tuple[pd.DataFrame, float, float]]:

        print (f"save_threshold= {save_threshold}")
        df = self.cleaning_dataframe(df)
        df, local_label_encoder_path = self.transform_data(df)
        df = self.select_features(df)
        df = self.scale_features(df)


        # train test split (maybe before scaling!)
        y = df["survived"]
        X = df.drop(columns="survived")

        # random state is set for the reproducibility's sake 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

        neighbours = 5
        knn = KNeighborsClassifier(n_neighbors=neighbours)
        knn.fit(X_train, y_train)

        y_train_pred = knn.predict(X_train)
        y_test_pred = knn.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        self.save_artifact(df, neighbours, train_accuracy, test_accuracy, X_train, y_train, knn, save_threshold, local_label_encoder_path)

        return (df, train_accuracy, test_accuracy)
    

    def preprocess_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.cleaning_dataframe(df)
        df, _ = self.transform_data(df, saveEncoders=False)
        df = self.select_features(df, full=False)
        df = self.scale_features(df)

        return df


    def predict(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            model_uri = f"models:/{MLPersist.MLFLOW_NAME}/{MLPersist.MLFLOW_ALIAS_STAGING}"
            loaded_model = mlflow.sklearn.load_model(model_uri) 
 
            print(loaded_model)
            y = loaded_model.predict(df)
            predictions_df = pd.DataFrame(y, columns=['prediction'], index=df.index)
            return predictions_df
        except Exception as e:
            print(f"MLflow: Error loading model: {e}")
            return None

    def test_predict(self, df_transformed: pd.DataFrame) -> Tuple[float, float]:
        y = df_transformed["survived"]
        X = df_transformed.drop(columns="survived")
        # random state is set for the reproducibility's sake 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

        y_train_pred = persist.predict(X_train)
        y_test_pred = persist.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"Train accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
        return (train_accuracy, test_accuracy)


if __name__ == "__main__":
    persist = MLPersist()

    file_url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
    df=pd.read_csv(file_url)

    df_t, _, _ = persist.train_pipeline(df.copy())
    persist.test_predict(df_t)

    df = df.drop(columns="survived")
    df = persist.preprocess_pipeline(df)
    print(persist.predict(df))

