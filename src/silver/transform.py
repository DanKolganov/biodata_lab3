import polars as pl
from deltalake import DeltaTable, write_deltalake
from pathlib import Path
import logging
from datetime import datetime
from typing import Optional

from ..config import Config

class SilverTransform:
    """Silver layer: Data cleaning and feature engineering"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.silver_path = Config.SILVER_PATH / "flights_clean"
        
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Config.LOGS_DIR / "silver_transform.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def load_bronze_data(self) -> pl.LazyFrame:
        """Load data from bronze layer"""
        self.logger.info("Loading data from bronze layer")
        
        # Use lazy scan for optimization
        df = pl.scan_delta(str(Config.BRONZE_PATH / "flights"))
        
        return df
    
    def clean_data(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Clean data: remove nulls, cancelled flights, outliers"""
        self.logger.info("Starting data cleaning")
        
        # Remove cancelled flights
        df = df.filter(pl.col("CANCELLED").eq(0))
        
        # Remove flights with missing critical fields
        critical_fields = ["ARR_DELAY", "DEP_DELAY", "DISTANCE", "AIR_TIME"]
        for field in critical_fields:
            if field in df.columns:
                df = df.filter(pl.col(field).is_not_null())
        
        # Remove outliers for delays (reasonable bounds)
        if "ARR_DELAY" in df.columns:
            df = df.filter(
                (pl.col("ARR_DELAY").ge(-120)) &  # Arrivals 2+ hours early
                (pl.col("ARR_DELAY").le(480))     # Arrivals 8+ hours late
            )
        
        if "DEP_DELAY" in df.columns:
            df = df.filter(
                (pl.col("DEP_DELAY").ge(-120)) &  # Departures 2+ hours early
                (pl.col("DEP_DELAY").le(480))     # Departures 8+ hours late
            )
        
        # Remove unrealistic distances and air times
        if "DISTANCE" in df.columns:
            df = df.filter(
                (pl.col("DISTANCE").ge(50)) &    # Minimum reasonable distance
                (pl.col("DISTANCE").le(5000))    # Maximum reasonable distance (miles)
            )
        
        if "AIR_TIME" in df.columns and "DISTANCE" in df.columns:
            # Remove unrealistic speed (distance/time > 1000 mph or < 100 mph)
            df = df.filter(
                (pl.col("DISTANCE") / pl.col("AIR_TIME") * 60).le(1000) &
                (pl.col("DISTANCE") / pl.col("AIR_TIME") * 60).ge(100)
            )
        
        self.logger.info("Data cleaning completed")
        return df
    
    def normalize_categories(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Normalize categorical fields"""
        self.logger.info("Normalizing categorical fields")
        
        # Standardize airline codes
        if "AIRLINE" in df.columns:
            df = df.with_columns([
                pl.col("AIRLINE").str.to_uppercase().str.strip_chars()
            ])
        
        # Standardize airport codes
        for airport_col in ["ORIGIN", "DEST"]:
            if airport_col in df.columns:
                df = df.with_columns([
                    pl.col(airport_col).str.to_uppercase().str.strip_chars()
                ])
        
        return df
    
    def add_derived_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Add derived features: hour, day_of_week, season, route"""
        self.logger.info("Adding derived features")
        
        # Extract hour from departure time
        if "CRS_DEP_TIME" in df.columns:
            df = df.with_columns([
                (pl.col("CRS_DEP_TIME") // 100).alias("DEP_HOUR").cast(pl.Int32)
            ])
        
        # Add day of week (if FL_DATE exists)
        if "FL_DATE" in df.columns:
            df = df.with_columns([
                pl.col("FL_DATE").str.to_datetime("%Y-%m-%d").dt.weekday().alias("DAY_OF_WEEK"),
                pl.col("FL_DATE").str.to_datetime("%Y-%m-%d").dt.month().alias("MONTH_NUM")
            ])
        
        # Add season based on month
        if "MONTH_NUM" in df.columns:
            df = df.with_columns([
                pl.when(pl.col("MONTH_NUM").is_in([12, 1, 2]))
                .then(pl.lit("Winter"))
                .when(pl.col("MONTH_NUM").is_in([3, 4, 5]))
                .then(pl.lit("Spring"))
                .when(pl.col("MONTH_NUM").is_in([6, 7, 8]))
                .then(pl.lit("Summer"))
                .when(pl.col("MONTH_NUM").is_in([9, 10, 11]))
                .then(pl.lit("Fall"))
                .otherwise(pl.lit("Unknown"))
                .alias("SEASON")
            ])
        
        # Add route (origin-dest)
        if "ORIGIN" in df.columns and "DEST" in df.columns:
            df = df.with_columns([
                (pl.col("ORIGIN") + "-" + pl.col("DEST")).alias("ROUTE")
            ])
        
        # Add delay categories
        if "ARR_DELAY" in df.columns:
            df = df.with_columns([
                pl.when(pl.col("ARR_DELAY").le(0))
                .then(pl.lit("On Time"))
                .when(pl.col("ARR_DELAY").le(15))
                .then(pl.lit("Minor Delay"))
                .when(pl.col("ARR_DELAY").le(60))
                .then(pl.lit("Moderate Delay"))
                .otherwise(pl.lit("Severe Delay"))
                .alias("DELAY_CATEGORY")
            ])
        
        return df
    
    def filter_by_year(self, df: pl.LazyFrame, year: int) -> pl.LazyFrame:
        """Filter data by specific year for batch processing"""
        return df.filter(pl.col("YEAR").eq(year))
    
    def select_relevant_columns(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Select relevant columns for analysis"""
        self.logger.info("Selecting relevant columns")
        
        # Core columns for analysis
        core_columns = [
            "YEAR", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "SEASON",
            "AIRLINE", "ORIGIN", "DEST", "ROUTE",
            "CRS_DEP_TIME", "DEP_HOUR", "CRS_ARR_TIME",
            "DEP_DELAY", "ARR_DELAY", "CANCELLED", "DIVERTED",
            "DISTANCE", "AIR_TIME",
            "DELAY_CATEGORY", "ingested_at"
        ]
        
        # Filter to only available columns
        available_columns = [col for col in core_columns if col in df.columns]
        
        return df.select(available_columns)
    
    def merge_with_existing(self, new_data: pl.LazyFrame) -> pl.LazyFrame:
        """Merge new data with existing using Delta Lake MERGE"""
        self.logger.info("Merging data with existing silver table")
        
        try:
            # Check if table exists
            dt = DeltaTable(str(self.silver_path))
            
            # Convert to pandas for merge operation
            new_pandas = new_data.collect().to_pandas()
            
            # Perform merge using DeltaTable
            from deltalake import DeltaMerger
            
            # Simple approach: overwrite for now
            # In production, implement proper MERGE logic
            return new_data
            
        except Exception as e:
            # Table doesn't exist, create new
            self.logger.info("Creating new silver table")
            return new_data
    
    def write_silver_data(self, df: pl.LazyFrame, mode: str = "overwrite") -> None:
        """Write transformed data to silver layer"""
        self.logger.info(f"Writing silver data with mode: {mode}")
        
        # Collect to pandas
        pandas_df = df.collect().to_pandas()
        
        # Write to Delta Lake with partitioning
        write_deltalake(
            table_or_uri=str(self.silver_path),
            data=pandas_df,
            mode=mode,
            partition_by=["YEAR", "MONTH"],
            engine="pyarrow"
        )
        
        self.logger.info(f"Successfully wrote {len(pandas_df)} records to silver layer")
    
    def transform_pipeline(self) -> None:
        """Complete silver transformation pipeline"""
        self.logger.info("Starting silver transformation pipeline")
        
        try:
            # Load bronze data
            bronze_df = self.load_bronze_data()
            
            # Apply transformations
            cleaned_df = self.clean_data(bronze_df)
            normalized_df = self.normalize_categories(cleaned_df)
            featured_df = self.add_derived_features(normalized_df)
            selected_df = self.select_relevant_columns(featured_df)
            
            # Merge with existing and write
            final_df = self.merge_with_existing(selected_df)
            self.write_silver_data(final_df, mode="overwrite")
            
            self.logger.info("Silver transformation pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in silver transformation pipeline: {str(e)}")
            raise
    
    def explain_query(self) -> str:
        """Explain query optimization for demonstration"""
        df = self.load_bronze_data()
        return df.explain()
