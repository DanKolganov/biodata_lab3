#!/usr/bin/env python3
"""
Lakehouse Pipeline for Flight Delay Analysis
Bronze → Silver → Gold → ML
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config import Config
from bronze.ingest import BronzeIngestion
from silver.transform import SilverTransform
from gold.aggregates import GoldAggregates
from ml.models import FlightDelayML
from delta_operations import DeltaOperations

class LakehousePipeline:
    """Complete lakehouse pipeline orchestrator"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        
        # Ensure directories exist
        Config.DATA_DIR.mkdir(exist_ok=True, parents=True)
        Config.LOGS_DIR.mkdir(exist_ok=True, parents=True)
        Config.MLFLOW_DIR.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.bronze = BronzeIngestion()
        self.silver = SilverTransform()
        self.gold = GoldAggregates()
        self.ml = FlightDelayML()
        self.delta_ops = DeltaOperations()
        
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Config.LOGS_DIR / "pipeline.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def run_bronze_layer(self) -> bool:
        """Run bronze layer ingestion"""
        try:
            self.logger.info("=== BRONZE LAYER ===")
            
            # Check if source file exists
            if not Config.SOURCE_CSV.exists():
                self.logger.error(f"Source file not found: {Config.SOURCE_CSV}")
                return False
            
            # Ingest data by years
            self.bronze.ingest_by_years(Config.SOURCE_CSV)
            
            # Get table info
            table_info = self.bronze.get_table_info()
            self.logger.info(f"Bronze table info: {table_info}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in bronze layer: {str(e)}")
            return False
    
    def run_silver_layer(self) -> bool:
        """Run silver layer transformation"""
        try:
            self.logger.info("=== SILVER LAYER ===")
            
            # Run transformation pipeline
            self.silver.transform_pipeline()
            
            # Show query optimization
            query_plan = self.silver.explain_query()
            self.logger.info(f"Query plan: {query_plan}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in silver layer: {str(e)}")
            return False
    
    def run_gold_layer(self) -> bool:
        """Run gold layer aggregation"""
        try:
            self.logger.info("=== GOLD LAYER ===")
            
            # Create all analytics
            self.gold.create_all_analytics()
            
            # Get table statistics
            stats = self.gold.get_table_stats()
            self.logger.info(f"Gold table stats: {stats}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in gold layer: {str(e)}")
            return False
    
    def run_ml_layer(self) -> bool:
        """Run ML modeling"""
        try:
            self.logger.info("=== ML LAYER ===")
            
            # Run ML pipeline
            results = self.ml.run_ml_pipeline()
            
            # Log results
            self.logger.info(f"Regression results: {len(results['regression'])} models")
            self.logger.info(f"Classification results: {len(results['classification'])} models")
            
            # Feature importance
            if results['feature_importance_regression']:
                top_features = list(results['feature_importance_regression'].keys())[:5]
                self.logger.info(f"Top regression features: {top_features}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in ML layer: {str(e)}")
            return False
    
    def run_delta_operations(self) -> bool:
        """Run Delta Lake operations"""
        try:
            self.logger.info("=== DELTA OPERATIONS ===")
            
            # Optimize all tables
            self.delta_ops.optimize_all_tables()
            
            # Demonstrate time travel
            time_travel_results = self.delta_ops.demonstrate_time_travel()
            self.logger.info(f"Time travel demo: {list(time_travel_results.keys())}")
            
            # Cleanup old data
            self.delta_ops.cleanup_old_data()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in Delta operations: {str(e)}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """Run complete lakehouse pipeline"""
        self.logger.info("Starting Lakehouse Pipeline for Flight Delay Analysis")
        
        # Run layers in sequence
        layers = [
            ("Bronze", self.run_bronze_layer),
            ("Silver", self.run_silver_layer),
            ("Gold", self.run_gold_layer),
            ("ML", self.run_ml_layer),
            ("Delta Operations", self.run_delta_operations)
        ]
        
        for layer_name, layer_func in layers:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"RUNNING {layer_name.upper()} LAYER")
            self.logger.info(f"{'='*50}")
            
            if not layer_func():
                self.logger.error(f"{layer_name} layer failed. Stopping pipeline.")
                return False
            
            self.logger.info(f"{layer_name} layer completed successfully")
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info("LAKEHOUSE PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info(f"{'='*50}")
        
        return True
    
    def run_specific_layer(self, layer: str) -> bool:
        """Run specific layer"""
        layer_map = {
            "bronze": self.run_bronze_layer,
            "silver": self.run_silver_layer,
            "gold": self.run_gold_layer,
            "ml": self.run_ml_layer,
            "delta": self.run_delta_operations
        }
        
        if layer.lower() not in layer_map:
            self.logger.error(f"Unknown layer: {layer}")
            return False
        
        return layer_map[layer.lower()]()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Lakehouse Pipeline for Flight Delay Analysis")
    parser.add_argument("--layer", choices=["bronze", "silver", "gold", "ml", "delta"], 
                       help="Run specific layer only")
    parser.add_argument("--full", action="store_true", help="Run full pipeline")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = LakehousePipeline()
    
    # Run pipeline
    if args.full or not args.layer:
        success = pipeline.run_full_pipeline()
    else:
        success = pipeline.run_specific_layer(args.layer)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
