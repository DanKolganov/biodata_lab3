import polars as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
from typing import Dict, Any

from config import Config

class SimpleFlightML:
    """Simplified ML models for flight delay prediction"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Config.LOGS_DIR / "simple_ml.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def load_features(self) -> pl.DataFrame:
        """Load feature table from CSV"""
        self.logger.info("Loading features from CSV")
        
        try:
            df = pl.read_csv(str(Config.GOLD_PATH / "features" / "features.csv"))
            self.logger.info(f"Loaded {len(df)} feature records")
            return df
        except Exception as e:
            self.logger.error(f"Error loading features: {str(e)}")
            # Fallback to direct CSV processing
            return self._create_features_from_source()
    
    def _create_features_from_source(self) -> pl.DataFrame:
        """Create features directly from source CSV"""
        self.logger.info("Creating features from source data")
        
        df = pl.scan_csv(str(Config.SOURCE_CSV))
        
        # Filter for 2024 and sample
        df = df.filter(pl.col("Year").eq(2024)).limit(10000)
        
        # Add derived features
        df = df.with_columns([
            (pl.col("CRSDepTime") // 100).alias("DEP_HOUR").cast(pl.Int32),
            pl.when(pl.col("Month").is_in([12, 1, 2]))
            .then(pl.lit("Winter"))
            .when(pl.col("Month").is_in([3, 4, 5]))
            .then(pl.lit("Spring"))
            .when(pl.col("Month").is_in([6, 7, 8]))
            .then(pl.lit("Summer"))
            .otherwise(pl.lit("Fall"))
            .alias("SEASON"),
            (pl.col("Origin") + "-" + pl.col("Dest")).alias("ROUTE")
        ])
        
        # Select features and filter
        feature_cols = [
            "Month", "DayofMonth", "DayOfWeek", "DEP_HOUR", "SEASON",
            "IATA_Code_Marketing_Airline", "Origin", "Dest", "ROUTE",
            "Distance", "DepDelay", "ArrDelay"
        ]
        
        return (df.select(feature_cols)
                .filter(
                    pl.col("ArrDelay").is_not_null() &
                    pl.col("DepDelay").is_not_null() &
                    pl.col("Distance").gt(0)
                )
                .with_columns([
                    pl.col("ArrDelay").alias("target_arrival_delay"),
                    (pl.col("ArrDelay") > 15).alias("target_is_delayed").cast(pl.Int32)
                ])
                .collect())
    
    def prepare_features(self, df: pl.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare features for ML modeling"""
        self.logger.info("Preparing features for ML")
        
        # Convert to pandas for sklearn
        pandas_df = df.to_pandas()
        
        # Define feature columns
        categorical_features = ['IATA_Code_Marketing_Airline', 'Origin', 'Dest', 'SEASON']
        numerical_features = ['Month', 'DayofMonth', 'DayOfWeek', 'DEP_HOUR', 'Distance', 'DepDelay']
        
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
        categorical_features = ['IATA_Code_Marketing_Airline', 'Origin', 'Dest', 'SEASON']
        numerical_features = ['Month', 'DayofMonth', 'DayOfWeek', 'DEP_HOUR', 'Distance', 'DepDelay']
        
        available_cat = [col for col in categorical_features if col in X.columns]
        available_num = [col for col in numerical_features if col in X.columns]
        
        preprocessor = self.create_preprocessor(available_cat, available_num)
        
        models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=Config.RANDOM_STATE)
        }
        
        results = {}
        
        for model_name, model in models.items():
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
        categorical_features = ['IATA_Code_Marketing_Airline', 'Origin', 'Dest', 'SEASON']
        numerical_features = ['Month', 'DayofMonth', 'DayOfWeek', 'DEP_HOUR', 'Distance', 'DepDelay']
        
        available_cat = [col for col in categorical_features if col in X.columns]
        available_num = [col for col in numerical_features if col in X.columns]
        
        preprocessor = self.create_preprocessor(available_cat, available_num)
        
        models = {
            'logistic_regression': LogisticRegression(random_state=Config.RANDOM_STATE, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=Config.RANDOM_STATE)
        }
        
        results = {}
        
        for model_name, model in models.items():
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
            num_features = [col for col in feature_names if col not in ['IATA_Code_Marketing_Airline', 'Origin', 'Dest', 'SEASON']]
            
            all_features = list(cat_features) + num_features
            
            # Create importance dictionary
            importance_dict = dict(zip(all_features, importances))
            
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return {}
    
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
    
    def show_results(self, results: Dict[str, Any]) -> None:
        """Display ML results summary"""
        print("\n" + "="*60)
        print("MACHINE LEARNING RESULTS SUMMARY")
        print("="*60)
        
        print(f"\nDataset Shape: {results['data_shape']}")
        print(f"Features: {len(results['feature_columns'])}")
        
        print(f"\nREGRESSION MODELS (Predicting Delay in Minutes):")
        for model_name, metrics in results['regression'].items():
            print(f"  {model_name}:")
            print(f"    RMSE: {metrics['rmse']:.2f} minutes")
            print(f"    R²: {metrics['r2']:.3f}")
        
        print(f"\nCLASSIFICATION MODELS (Predicting Delay > 15 min):")
        for model_name, metrics in results['classification'].items():
            print(f"  {model_name}:")
            print(f"    Accuracy: {metrics['accuracy']:.3f}")
            print(f"    ROC AUC: {metrics['roc_auc']:.3f}")
        
        print(f"\nTOP 10 FEATURE IMPORTANCE (Regression):")
        for i, (feature, importance) in enumerate(list(results['feature_importance_regression'].items())[:10]):
            print(f"  {i+1:2d}. {feature:25s}: {importance:.4f}")
        
        print("\n" + "="*60)
