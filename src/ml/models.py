import polars as pl
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
from typing import Dict, Tuple, Any

from config import Config

class FlightDelayML:
    """ML models for flight delay prediction with MLflow logging"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.setup_mlflow()
        
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Config.LOGS_DIR / "ml_models.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def setup_mlflow(self):
        """Setup MLflow connection"""
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment("flight_delay_prediction")
        
    def load_features(self) -> pl.DataFrame:
        """Load feature table from gold layer"""
        self.logger.info("Loading features from gold layer")
        
        try:
            df = pl.scan_delta(str(Config.GOLD_PATH / "features"))
            return df.collect()
        except Exception as e:
            self.logger.error(f"Error loading features: {str(e)}")
            raise
    
    def prepare_features(self, df: pl.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare features for ML modeling"""
        self.logger.info("Preparing features for ML")
        
        # Convert to pandas for sklearn
        pandas_df = df.to_pandas()
        
        # Define feature columns
        categorical_features = ['AIRLINE', 'ORIGIN', 'DEST', 'DAY_OF_WEEK', 'DEP_HOUR']
        numerical_features = ['DISTANCE', 'AIR_TIME', 'DEP_DELAY', 'MONTH', 'DAY_OF_MONTH']
        
        # Ensure columns exist
        available_cat = [col for col in categorical_features if col in pandas_df.columns]
        available_num = [col for col in numerical_features if col in pandas_df.columns]
        
        # Drop rows with missing values
        feature_cols = available_cat + available_num
        target_cols = ['target_arrival_delay', 'target_is_delayed']
        
        pandas_df = pandas_df[feature_cols + target_cols].dropna()
        
        # Separate features and targets
        X = pandas_df[feature_cols]
        y_regression = pandas_df['target_arrival_delay']
        y_classification = pandas_df['target_is_delayed']
        
        return X, y_regression, y_classification
    
    def create_preprocessor(self, categorical_features: list, numerical_features: list) -> ColumnTransformer:
        """Create preprocessing pipeline"""
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )
        return preprocessor
    
    def train_regression_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train regression models for arrival delay prediction"""
        self.logger.info("Training regression models")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
        )
        
        # Define features
        categorical_features = ['AIRLINE', 'ORIGIN', 'DEST', 'DAY_OF_WEEK', 'DEP_HOUR']
        numerical_features = ['DISTANCE', 'AIR_TIME', 'DEP_DELAY', 'MONTH', 'DAY_OF_MONTH']
        
        available_cat = [col for col in categorical_features if col in X.columns]
        available_num = [col for col in numerical_features if col in X.columns]
        
        preprocessor = self.create_preprocessor(available_cat, available_num)
        
        models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=Config.RANDOM_STATE)
        }
        
        results = {}
        
        for model_name, model in models.items():
            with mlflow.start_run(run_name=f"regression_{model_name}"):
                # Create pipeline
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', model)
                ])
                
                # Train model
                pipeline.fit(X_train, y_train)
                
                # Predictions
                y_pred = pipeline.predict(X_test)
                
                # Metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                # Log metrics
                mlflow.log_metrics({
                    'mse': mse,
                    'rmse': rmse,
                    'r2_score': r2
                })
                
                # Log model
                mlflow.sklearn.log_model(pipeline, f"model_{model_name}")
                
                # Log parameters
                mlflow.log_params({
                    'model_type': 'regression',
                    'model_name': model_name,
                    'test_size': Config.TEST_SIZE,
                    'random_state': Config.RANDOM_STATE,
                    'features': list(X.columns),
                    'gold_table_version': self.get_gold_table_version()
                })
                
                results[model_name] = {
                    'model': pipeline,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'predictions': y_pred
                }
                
                self.logger.info(f"{model_name} - RMSE: {rmse:.2f}, R2: {r2:.3f}")
        
        return results
    
    def train_classification_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train classification models for delay prediction"""
        self.logger.info("Training classification models")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
        )
        
        # Define features
        categorical_features = ['AIRLINE', 'ORIGIN', 'DEST', 'DAY_OF_WEEK', 'DEP_HOUR']
        numerical_features = ['DISTANCE', 'AIR_TIME', 'DEP_DELAY', 'MONTH', 'DAY_OF_MONTH']
        
        available_cat = [col for col in categorical_features if col in X.columns]
        available_num = [col for col in numerical_features if col in X.columns]
        
        preprocessor = self.create_preprocessor(available_cat, available_num)
        
        models = {
            'logistic_regression': LogisticRegression(random_state=Config.RANDOM_STATE, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=Config.RANDOM_STATE)
        }
        
        results = {}
        
        for model_name, model in models.items():
            with mlflow.start_run(run_name=f"classification_{model_name}"):
                # Create pipeline
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', model)
                ])
                
                # Train model
                pipeline.fit(X_train, y_train)
                
                # Predictions
                y_pred = pipeline.predict(X_test)
                y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                # Log metrics
                mlflow.log_metrics({
                    'accuracy': accuracy,
                    'roc_auc': roc_auc
                })
                
                # Log model
                mlflow.sklearn.log_model(pipeline, f"model_{model_name}")
                
                # Log parameters
                mlflow.log_params({
                    'model_type': 'classification',
                    'model_name': model_name,
                    'test_size': Config.TEST_SIZE,
                    'random_state': Config.RANDOM_STATE,
                    'delay_threshold': Config.DELAY_THRESHOLD_MINUTES,
                    'features': list(X.columns),
                    'gold_table_version': self.get_gold_table_version()
                })
                
                results[model_name] = {
                    'model': pipeline,
                    'accuracy': accuracy,
                    'roc_auc': roc_auc,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                self.logger.info(f"{model_name} - Accuracy: {accuracy:.3f}, ROC AUC: {roc_auc:.3f}")
        
        return results
    
    def get_feature_importance(self, model: Pipeline, feature_names: list) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if hasattr(model.named_steps['model'], 'feature_importances_'):
            importances = model.named_steps['model'].feature_importances_
            
            # Get feature names after preprocessing
            preprocessor = model.named_steps['preprocessor']
            cat_features = []
            num_features = []
            
            # Handle categorical features (one-hot encoded)
            if hasattr(preprocessor, 'named_transformers_'):
                cat_encoder = preprocessor.named_transformers_['cat']
                if hasattr(cat_encoder, 'get_feature_names_out'):
                    cat_features = cat_encoder.get_feature_names_out()
            
            # Handle numerical features
            num_features = [col for col in feature_names if col not in ['AIRLINE', 'ORIGIN', 'DEST', 'DAY_OF_WEEK', 'DEP_HOUR']]
            
            all_features = list(cat_features) + num_features
            
            # Create importance dictionary
            importance_dict = dict(zip(all_features, importances))
            
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return {}
    
    def get_gold_table_version(self) -> int:
        """Get current gold table version for lineage"""
        try:
            from deltalake import DeltaTable
            dt = DeltaTable(str(Config.GOLD_PATH / "features"))
            return dt.version()
        except:
            return 0
    
    def run_ml_pipeline(self) -> Dict[str, Any]:
        """Run complete ML pipeline"""
        self.logger.info("Starting ML pipeline")
        
        try:
            # Load and prepare data
            df = self.load_features()
            X, y_reg, y_clf = self.prepare_features(df)
            
            # Train models
            regression_results = self.train_regression_models(X, y_reg)
            classification_results = self.train_classification_models(X, y_clf)
            
            # Get feature importance for best models
            best_reg_model = regression_results['random_forest']['model']
            best_clf_model = classification_results['random_forest']['model']
            
            reg_importance = self.get_feature_importance(best_reg_model, list(X.columns))
            clf_importance = self.get_feature_importance(best_clf_model, list(X.columns))
            
            results = {
                'regression': regression_results,
                'classification': classification_results,
                'feature_importance_regression': reg_importance,
                'feature_importance_classification': clf_importance,
                'data_shape': df.shape,
                'feature_columns': list(X.columns)
            }
            
            self.logger.info("ML pipeline completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in ML pipeline: {str(e)}")
            raise
