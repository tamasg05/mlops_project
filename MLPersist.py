import pickle
import pandas as pd
from scipy import stats
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import os

class MLPersist:
    MODEL_FOLDER = "artifacts/"
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

    def transfrom_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # encoding the "Deck" column
        col = 'deck'
        transformed_as_df = pd.get_dummies(df[col])
        coded_column_names = [col + "_" + column for column in transformed_as_df.columns]
        transformed_as_df.columns = coded_column_names

        df = pd.concat([df, transformed_as_df], axis=1)

        # encoding adult_male and alone columns    
        label_encoders = {}
        for col in ["adult_male", "alone"]:
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col])

        return df

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = ['pclass', 'adult_male', 'alone', 'fare', 'deck_A', 'deck_B', 'deck_C', 'deck_D', 'deck_E', 'deck_F', 'deck_G', 'deck_None', 'survived']
        df = df[features]
        return df

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Normalization
        min_max_scaler = MinMaxScaler(feature_range=(0, 10))
        col = 'fare'
        df[col] = min_max_scaler.fit_transform(df[[col]])
        return df
    
    def train_pipeline(self) -> None:
        file_url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
        df=pd.read_csv(file_url)

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

        self.save_artifact(knn, MLPersist.MODEL_FOLDER + "knn_classifier.pkl")

        y_train_pred = knn.predict(X_train)
        y_test_pred = knn.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"Train accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")


if __name__ == "__main__":
    persist = MLPersist()

    persist.train_pipeline()