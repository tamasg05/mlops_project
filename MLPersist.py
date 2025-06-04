import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.entities.model_registry import ModelVersion

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

import os
from typing import Optional, Tuple
from datetime import datetime
from pathlib import Path


class MLPersist:
    MODEL_FOLDER = "artifacts/"
    LABEL_ENCODER = "label_encoder_dict.pkl"
    KNN_MODEL = "knn_classifier.pkl"

    MLFLOW_NAME = "Titanic_KNN_Model"

    MLFLOW_ALIAS_STAGING = "Staging"
    MLFLOW_ALIAS_PRODUCTION = "Production"


    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Titanic KNN classification tests")

    def __init__(self):
        if os.path.exists(MLPersist.MODEL_FOLDER):
            print(MLPersist.MODEL_FOLDER + " exists.")
        else:
            Path(MLPersist.MODEL_FOLDER).mkdir(parents=True, exist_ok=True)

    def save_artifact(self, artifact, file_path) -> None:
        with open(file_path, 'wb') as file:
            pickle.dump(artifact, file)

    def load_artifact(self, file_path) -> any:
        with open(file_path, 'rb') as file:
            artifact = pickle.load(file)
        return artifact

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

    def transfrom_data(self, df: pd.DataFrame, saveEncoders=True) -> pd.DataFrame:
        # encoding the "Deck" column
        col = 'deck'
        transformed_as_df = pd.get_dummies(df[col])
        coded_column_names = [col + "_" + column for column in transformed_as_df.columns]
        transformed_as_df.columns = coded_column_names

        df = pd.concat([df, transformed_as_df], axis=1)

        # encoding adult_male and alone columns    
        if saveEncoders: 
            label_encoders = {}
            for col in ["adult_male", "alone"]:
                label_encoders[col] = LabelEncoder()
                df[col] = label_encoders[col].fit_transform(df[col])
            
            self.save_artifact(label_encoders, MLPersist.MODEL_FOLDER + MLPersist.LABEL_ENCODER)
        else:
            label_encoders = self.load_artifact(MLPersist.MODEL_FOLDER + MLPersist.LABEL_ENCODER)
            for col in ["adult_male", "alone"]:
                df[col] = label_encoders[col].transform(df[col])

        return df

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
        df = self.transfrom_data(df)
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
        
        ######
        # TODO saving training data
        train_data_path = os.path.join(self.MODEL_FOLDER, "training_data.csv")
        df.to_csv(train_data_path, index=False) # Save the DataFrame to a CSV file
        print(f"Temporary training data saved to: {train_data_path}")
        ######

        try:
            # MLflow integration for saving the model
            # Setting a unique name for the run
            rname = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with mlflow.start_run(run_name=rname) as run:
                mlflow.log_param("Neighbors for KNN", neighbours)
                mlflow.log_metric("train_accuracy", train_accuracy)
                mlflow.log_metric("test_accuracy", test_accuracy)
                
                mlflow.log_artifact(train_data_path, artifact_path="training_data")

                # setting the input data shapes
                # TODO convert int columns to float to avoid the possible NaN waring in MLFlow
                signature = mlflow.models.infer_signature(X_train, y_train)



                if test_accuracy > save_threshold:
                    mlflow.sklearn.log_model(sk_model=knn, 
                                            artifact_path="knn_model",
                                            signature=signature)
                    
                    model_uri = f"runs:/{run.info.run_id}/knn_model"
                    registered_model_version = mlflow.register_model(model_uri=model_uri, name=MLPersist.MLFLOW_NAME)
                                
                    # Transition model version to "Staging"
                    mlflow_client = mlflow.tracking.MlflowClient()

                    try:
                        # TODO deprecated: change
                        current_staging_versions = mlflow_client.get_latest_versions(
                            MLPersist.MLFLOW_NAME
                        )
                    except Exception as e:
                        print(f"Unexpected error while getting the existing versions from {MLPersist.MLFLOW_ALIAS_STAGING}: {e}")


                    # Assign the 'Staging' alias to the newly registered model version
                    mlflow_client.set_registered_model_alias(
                        name=MLPersist.MLFLOW_NAME,
                        alias=MLPersist.MLFLOW_ALIAS_STAGING, # Use the defined alias constant
                        version=registered_model_version.version
                    )
                    print(f"MLflow: Model Name: {MLPersist.MLFLOW_NAME}, Model Version: {registered_model_version.version} registered and aliased: '{MLPersist.MLFLOW_ALIAS_STAGING}'.")               
                    print("MLflow: Trained model saved.")

            print(f"Train accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

        except Exception as e:
            print(f"Failed saving the model with the exception: {str(e)}")
            None

        return (df, train_accuracy, test_accuracy)
    

    def get_model_in_stage(self, stage: str, model_name=None) -> Optional[ModelVersion]:
        if model_name is None:
            model_name = MLPersist.MLFLOW_NAME

        mlflow_client = mlflow.tracking.MlflowClient()
        try:
            # TODO deprecated: change
            model_version = mlflow_client.get_latest_versions(model_name, stages=[stage])

            if model_version:
                # get_latest_versions returns a list, even if typically one for a stage
                return model_version[0]
            else:
                print(f"No model found for '{model_name}' in stage '{stage}'.")
                return None
        except Exception as e:
            print(f"Error checking model in stage: {e}")
            return None

    def preprocess_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.cleaning_dataframe(df)
        df = self.transfrom_data(df, saveEncoders=False)
        df = self.select_features(df, full=False)
        df = self.scale_features(df)

        return df


    def predict(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        # MLflow integration for loading the latest model from the given stage
        
        try:
            model_uri = f"models:/{MLPersist.MLFLOW_NAME}/{MLPersist.MLFLOW_ALIAS_PRODUCTION}"
            loaded_model = mlflow.sklearn.load_model(model_uri) # Loads the latest saved model
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

    # file_url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
    # df=pd.read_csv(file_url)

    # df_t, _, _ = persist.train_pipeline(df.copy())
    # persist.test_predict(df_t)

    # df = df.drop(columns="survived")
    # df = persist.preprocess_pipeline(df)
    # print(persist.predict(df))

    print(persist.get_model_in_stage(MLPersist.MLFLOW_ALIAS_STAGING))