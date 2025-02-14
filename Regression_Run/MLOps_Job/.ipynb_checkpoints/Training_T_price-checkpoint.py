import pandas as pd
import xgboost as xgb
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV

# Data Preprocessor Class
class DataPreprocessor:
    def __init__(self, df):
        self.df = df.copy()
    
    def fill_missing_values(self):
        numeric_columns = ['Trip_Distance_km', 'Palette_Count', 'Base_Fare', 'Per_Km_Rate', 
                           'Per_Minute_Rate', 'Trip_Duration_Minutes', 'Trip_Price']
        categorical_columns = ['Starting_Day', 'Day_of_Week', 'Load_Count', 'Weather']
        
        self.df[numeric_columns] = self.df[numeric_columns].fillna(self.df[numeric_columns].median())
        self.df[categorical_columns] = self.df[categorical_columns].fillna(self.df[categorical_columns].mode().iloc[0])
    
    def one_hot_encode(self):
        categorical_columns = ['Starting_Day', 'Day_of_Week', 'Load_Count', 'Weather']
        self.df = pd.get_dummies(self.df, columns=categorical_columns, drop_first=True)
    
    def preprocess(self):
        self.fill_missing_values()
        self.one_hot_encode()
        return self.df

# Model Training Class
class ModelTrainer:
    def __init__(self, df):
        self.df = df
        self.X = self.df.drop(columns=['Trip_Price'])
        self.y = self.df['Trip_Price']
        self.scaler = RobustScaler()
        self.model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
    
    def prepare_data(self):
        numeric_columns = ['Trip_Distance_km', 'Palette_Count', 'Base_Fare', 'Per_Km_Rate', 'Per_Minute_Rate', 'Trip_Duration_Minutes']
        self.X[numeric_columns] = self.scaler.fit_transform(self.X[numeric_columns])
        return train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        param_grid = {'n_estimators': [100], 'learning_rate': [0.1], 'max_depth': [5], 
                      'subsample': [0.8], 'min_child_weight': [1]}
        
        grid_search = GridSearchCV(self.model, param_grid, cv=3, n_jobs=-1, error_score='raise')
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_estimator_, self.scaler, X_train.columns

if __name__ == "__main__":
    df = pd.read_csv("data/Truck_price_prediction.csv")
    preprocessor = DataPreprocessor(df)
    df_cleaned = preprocessor.preprocess()
    
    trainer = ModelTrainer(df_cleaned)
    model, scaler, feature_names = trainer.train_model()

    # Save the trained model and scaler
    joblib.dump(model, "Model_objects/xgb_model.pkl")
    joblib.dump(scaler, "Model_objects/scaler.pkl")
    joblib.dump(feature_names, "Model_objects/feature_names.pkl")

    print("Model training complete. Files saved.")
