import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

from evidently import Report
from evidently.presets import DataDriftPreset




class MLPersist:
    MODEL_FOLDER = "artifacts/"
    LABEL_ENCODER = "label_encoder_dict.pkl"
    LABEL_ENCODER_FOLDER = "label_encoder_dictionary"
    MLFLOW_NAME = "Titanic_KNN_Model"

    DRIFT_REPORT_PATH = MODEL_FOLDER + "data_drift_report.html"

    MLFLOW_ALIAS_STAGING = "Staging"
    MLFLOW_ALIAS_PRODUCTION = "Production"
    MLFLOW_ALIAS_TEST = "Test"

    def __init__(self):
        
        self.last_orig_df = None

        # Set up MLflow tracking
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        experiment_name = "Titanic KNN classification tests"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

        # Ensure artifact folder exists
        if os.path.exists(MLPersist.MODEL_FOLDER):
            print(f"{MLPersist.MODEL_FOLDER} exists.")
        else:
            Path(MLPersist.MODEL_FOLDER).mkdir(parents=True, exist_ok=True)

    def generate_data_drift_report(self, reference_df: pd.DataFrame, current_df: pd.DataFrame, output_path: str = DRIFT_REPORT_PATH):
        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(reference_data=reference_df, current_data=current_df)
        snapshot.save_html(output_path) 
        print(f"Data drift report saved to {output_path}")
        return output_path


    def save_artifact(
        self,
        df_cleaned_transformed: pd.DataFrame,
        df_orig: pd.DataFrame,
        neighbours: int,
        train_accuracy: float,
        test_accuracy: float,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        knn: KNeighborsClassifier,
        save_threshold: float,
        label_encoder_path: str
    ) -> None:
        """Save the model, encoders, and training data as MLflow artifacts."""
        clean_transformed_training_data_path = os.path.join(self.MODEL_FOLDER, "cleaned_transformed_training_data.csv")
        df_cleaned_transformed.to_csv(clean_transformed_training_data_path, index=False)
        print(f"Temporary cleaned, transformed training data saved to: {clean_transformed_training_data_path}")

        orig_training_data_path = os.path.join(self.MODEL_FOLDER, "orig_training_data.csv")
        df_orig.to_csv(orig_training_data_path, index=False)
        print(f"Original training data saved to: {orig_training_data_path}")

        mlflow_client = mlflow.tracking.MlflowClient()

        # Ensure model exists in registry
        try:
            mlflow_client.get_registered_model(name=MLPersist.MLFLOW_NAME)
        except mlflow.exceptions.RestException:
            mlflow_client.create_registered_model(name=MLPersist.MLFLOW_NAME)

        try:
            run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_param("Neighbors for KNN", neighbours)
                mlflow.log_metric("train_accuracy", train_accuracy)
                mlflow.log_metric("test_accuracy", test_accuracy)

                # df_cleaned_transformed: Fixing schema warning for type inference: convert int columns to float64 to support NaNs
                df_cleaned_transformed = df_cleaned_transformed.astype({
                    col: 'float64' for col in df_cleaned_transformed.select_dtypes(include='int').columns
                })
                mlflow.log_input(mlflow.data.from_pandas(df_cleaned_transformed), context="training")
                
                mlflow.log_artifact(clean_transformed_training_data_path, artifact_path="training_data")
                mlflow.log_artifact(orig_training_data_path, artifact_path="training_data")
                mlflow.log_artifact(label_encoder_path, artifact_path=MLPersist.LABEL_ENCODER_FOLDER)

                # X_train, y_train: Fixing schema warning for  type inference for infer_signature(): 
                # converting int columns to float64 to support NaNs
                X_train = X_train.astype({
                    col: 'float64' for col in X_train.select_dtypes(include='int').columns
                })
                y_train = y_train.astype('float64')

                signature = mlflow.models.infer_signature(X_train, y_train)
                input_example = X_train.head(1)

                # this logs the model AND registers a new version
                mlflow.sklearn.log_model(
                    sk_model=knn,
                    name=MLPersist.MLFLOW_NAME,
                    registered_model_name=MLPersist.MLFLOW_NAME,
                    signature=signature,
                    input_example=input_example
                )

            # after the run ends, we retrieve the latest version from the run_id
            versions = mlflow_client.search_model_versions(
                f"name='{MLPersist.MLFLOW_NAME}' and run_id='{run.info.run_id}'"
            )
            if not versions:
                raise RuntimeError("Could not find the registered model version after logging.")
            version = versions[0].version

            if test_accuracy > save_threshold:
                mlflow_client.set_registered_model_alias(MLPersist.MLFLOW_NAME, MLPersist.MLFLOW_ALIAS_TEST, version)
                mlflow_client.set_registered_model_alias(MLPersist.MLFLOW_NAME, MLPersist.MLFLOW_ALIAS_STAGING, version)
                print(f"Model v{version} aliased as Staging and Test.")
            else:
                mlflow_client.set_registered_model_alias(MLPersist.MLFLOW_NAME, MLPersist.MLFLOW_ALIAS_TEST, version)
                print(f"Model v{version} aliased as Test only.")

            print(f"Train accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

        except Exception as e:
            print(f"Failed saving the model: {e}")
            raise

    def save_artifact_to_disk(self, artifact: any, file_path: str) -> None:
        with open(file_path, 'wb') as file:
            pickle.dump(artifact, file)

    def load_artifact_from_disk(self, file_path: str) -> any:
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def load_label_encoders(self, alias: str) -> Dict[str, LabelEncoder]:
        """Load label encoders from MLflow by alias."""
        try:
            client = mlflow.tracking.MlflowClient()
            # Get model version associated with alias
            model_version = client.get_model_version_by_alias(name=self.MLFLOW_NAME, alias=alias)
            run_id = model_version.run_id
            artifact_uri = f"runs:/{run_id}/{self.LABEL_ENCODER_FOLDER}/{self.LABEL_ENCODER}"
            print(f"Loading label encoders from MLflow URI: {artifact_uri}")

            local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
            with open(local_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Failed to load label encoders from MLflow: {e}")
            raise

    def cleaning_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill or drop missing values and clean up column types."""
        df['deck'] = df['deck'].astype('category')
        df['deck'] = df['deck'].cat.add_categories("None").fillna("None")
        df['age'] = df['age'].fillna(df['age'].median())
        return df.dropna()

    def transform_data(
        self,
        df: pd.DataFrame,
        alias: str = MLFLOW_ALIAS_STAGING,
        save_encoders: bool = True
    ) -> Tuple[pd.DataFrame, str]:
        """One-hot and label encode columns. Optionally save/load encoders."""

        # encoding the "Deck" column
        transformed_as_df = pd.get_dummies(df['deck'])
        transformed_as_df.columns = [f"deck_{col}" for col in transformed_as_df.columns]
        df = pd.concat([df, transformed_as_df], axis=1)

        encoder_path = self.MODEL_FOLDER + self.LABEL_ENCODER

        # encoding adult_male and alone columns
        if save_encoders:
            encoders = {}
            for col in ["adult_male", "alone"]:
                encoders[col] = LabelEncoder()
                df[col] = encoders[col].fit_transform(df[col])
            self.save_artifact_to_disk(encoders, encoder_path)
        else:
            encoders = self.load_label_encoders(alias)
            for col in ["adult_male", "alone"]:
                df[col] = encoders[col].transform(df[col])

        return df, encoder_path

    def select_features(self, df: pd.DataFrame, full: bool = True) -> pd.DataFrame:
        features = [
            'pclass', 'adult_male', 'alone', 'fare',
            'deck_A', 'deck_B', 'deck_C', 'deck_D', 'deck_E',
            'deck_F', 'deck_G', 'deck_None'
        ]
        if full:
            features.append('survived')
        return df[features]

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize the 'fare' column to range [0, 10]."""
        scaler = MinMaxScaler(feature_range=(0, 10))
        df['fare'] = scaler.fit_transform(df[['fare']])
        return df

    def train_pipeline(
        self,
        df: pd.DataFrame,
        save_threshold: float = 0.0
    ) -> Optional[Tuple[pd.DataFrame, float, float]]:
        """Train the KNN model and log it to MLflow if criteria are met."""

        # keeping it for data drift check
        self.last_orig_df = df.copy()

        df_orig = df.copy()
        print(f"save_threshold= {save_threshold}")
        df = self.cleaning_dataframe(df)
        df, encoder_path = self.transform_data(df)
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
        
        self.save_artifact(df, df_orig, neighbours, train_accuracy, test_accuracy, X_train, y_train, knn, save_threshold, encoder_path)

        return (df, train_accuracy, test_accuracy)
    

    def preprocess_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for prediction."""
        df = self.cleaning_dataframe(df)
        df, _ = self.transform_data(df, save_encoders=False)
        df = self.select_features(df, full=False)
        return self.scale_features(df)

    def predict(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Predict using the latest staged model."""
        try:
            model_uri = f"models:/{self.MLFLOW_NAME}@{self.MLFLOW_ALIAS_STAGING}"
            model = mlflow.sklearn.load_model(model_uri)
            predictions = model.predict(df)
            return pd.DataFrame(predictions, columns=['prediction'], index=df.index)
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def test_predict(self, df_transformed: pd.DataFrame) -> Tuple[float, float]:
        """Evaluate prediction accuracy using the staged model."""
        y = df_transformed["survived"]
        X = df_transformed.drop(columns="survived")
        # random state is set for the reproducibility's sake 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

        y_train_pred = self.predict(X_train)['prediction']
        y_test_pred = self.predict(X_test)['prediction']

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"Train accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
        return (train_accuracy, test_accuracy)


if __name__ == "__main__":
    persist = MLPersist()

    file_url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
    df = pd.read_csv(file_url)

    df_t, _, _ = persist.train_pipeline(df.copy())
    persist.test_predict(df_t)

    df = df.drop(columns="survived")
    df = persist.preprocess_pipeline(df)
    print(persist.predict(df))
