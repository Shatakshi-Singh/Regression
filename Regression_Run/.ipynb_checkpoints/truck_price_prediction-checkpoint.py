import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

class DataPreprocessor:
    
    def __init__(self, df):
        self.df = df.copy()
    
    def fill_missing_values(self):

        numeric_columns = ['Trip_Distance_km', 'Palette_Count', 'Base_Fare',
                           'Per_Km_Rate', 'Per_Minute_Rate', 'Trip_Duration_Minutes', 'Trip_Price']
        categorical_columns = ['Starting_Day', 'Day_of_Week', 'Load_Count', 'Weather']
        
        # median
        self.df[numeric_columns] = self.df[numeric_columns].fillna(self.df[numeric_columns].median())

        # mode
        self.df[categorical_columns] = self.df[categorical_columns].fillna(self.df[categorical_columns].mode().iloc[0])
    
    def one_hot_encode(self):
        categorical_columns = ['Starting_Day', 'Day_of_Week', 'Load_Count', 'Weather']
        self.df = pd.get_dummies(self.df, columns=categorical_columns, drop_first=True)
        #debugging1
        # print("Columns after One-Hot Encoding:", self.df.columns)
    
    def preprocess(self):
        self.fill_missing_values()
        self.one_hot_encode()
        return self.df


class ModelTrainEval:
    
    def __init__(self, df):
        self.df = df
        self.X = None
        self.y = None
        self.scaler = RobustScaler()
        self.model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
    
    def prepare_data(self):
        self.X = self.df.drop(columns=['Trip_Price'])
        self.y = self.df['Trip_Price']

        numeric_columns = ['Trip_Distance_km', 'Palette_Count', 'Base_Fare', 'Per_Km_Rate', 
                           'Per_Minute_Rate', 'Trip_Duration_Minutes']
        
        self.X[numeric_columns] = self.scaler.fit_transform(self.X[numeric_columns])
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Debugging
        # print(f"NaN values in X_train:\n{X_train.isnull().sum()}\n")
        # print(f"NaN values in X_test:\n{X_test.isnull().sum()}\n")
        # print(f"NaN values in y_train: {y_train.isnull().sum()}")
        # print(f"NaN values in y_test: {y_test.isnull().sum()}")

        return train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()

        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.1, 0.2],
            'max_depth': [3, 5, 10],
            'subsample': [0.8, 1.0],
            'min_child_weight': [1, 3, 5]
        }

        grid_search = GridSearchCV(self.model, param_grid, cv=5, n_jobs=-1,error_score='raise')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        train_score = round(best_model.score(X_train, y_train), 4)
        test_score = round(best_model.score(X_test, y_test), 4)

        # print("Best Model:", best_model)
        return train_score, test_score


if __name__ == "__main__":
    df = pd.read_csv("Truck_price_prediction.csv")
    preprocessor = DataPreprocessor(df)
    df_cleaned = preprocessor.preprocess()
   
    trainer = ModelTrainEval(df_cleaned)
    train_score, test_score = trainer.train_model()
    print(f"Train Score: {train_score}, Test Score: {test_score}")
