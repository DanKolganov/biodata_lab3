import polars as pl
from deltalake import DeltaTable, write_deltalake
import logging
from datetime import datetime
from typing import Optional

from config import Config

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
        
        # For now, use direct CSV reading to avoid Delta Lake compatibility issues
        # In production, this would read from bronze Delta table
        df = pl.scan_csv(str(Config.SOURCE_CSV))
        
        # Filter to specific year for demonstration
        df = df.filter(pl.col("Year").eq(2024))
        
        return df
    
    def clean_data(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Clean data: remove nulls, cancelled flights, outliers"""
        self.logger.info("Starting data cleaning")
        
        # Remove cancelled flights
        df = df.filter(pl.col("Cancelled").eq(0))
        
        # Remove flights with missing critical fields
        critical_fields = ["ArrDelay", "DepDelay", "Distance", "AirTime"]
        for field in critical_fields:
            if field in df.columns:
                df = df.filter(pl.col(field).is_not_null())
        
        # Remove outliers for delays (reasonable bounds)
        if "ArrDelay" in df.columns:
            df = df.filter(
                (pl.col("ArrDelay").ge(-120)) &  # Arrivals 2+ hours early
                (pl.col("ArrDelay").le(480))     # Arrivals 8+ hours late
            )
        
        if "DepDelay" in df.columns:
            df = df.filter(
                (pl.col("DepDelay").ge(-120)) &  # Departures 2+ hours early
                (pl.col("DepDelay").le(480))     # Departures 8+ hours late
            )
        
        # Remove unrealistic distances and air times
        if "Distance" in df.columns:
            df = df.filter(
                (pl.col("Distance").ge(50)) &    # Minimum reasonable distance
                (pl.col("Distance").le(5000))    # Maximum reasonable distance (miles)
            )
        
        if "AirTime" in df.columns and "Distance" in df.columns:
            # Remove unrealistic speed (distance/time > 1000 mph or < 100 mph)
            df = df.filter(
                (pl.col("Distance") / pl.col("AirTime") * 60).le(1000) &
                (pl.col("Distance") / pl.col("AirTime") * 60).ge(100)
            )
        
        self.logger.info("Data cleaning completed")
        return df
    
    def normalize_categories(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Normalize categorical fields"""
        self.logger.info("Normalizing categorical fields")
        
        # Standardize airline codes
        if "IATA_Code_Marketing_Airline" in df.columns:
            df = df.with_columns([
                pl.col("IATA_Code_Marketing_Airline").str.to_uppercase().str.strip_chars().alias("AIRLINE")
            ])
        
        # Standardize airport codes
        for airport_col in ["Origin", "Dest"]:
            if airport_col in df.columns:
                df = df.with_columns([
                    pl.col(airport_col).str.to_uppercase().str.strip_chars()
                ])
        
        return df
    
    def add_derived_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Add derived features: hour, day_of_week, season, route"""
        self.logger.info("Adding derived features")
        
        # Extract hour from departure time
        if "CRSDepTime" in df.columns:
            df = df.with_columns([
                (pl.col("CRSDepTime") // 100).alias("DEP_HOUR").cast(pl.Int32)
            ])
        
        # Add day of week (already exists as DayOfWeek)
        if "DayOfWeek" in df.columns:
            df = df.with_columns([
                pl.col("DayOfWeek").alias("DAY_OF_WEEK")
            ])
        
        # Add season based on month
        if "Month" in df.columns:
            df = df.with_columns([
                pl.when(pl.col("Month").is_in([12, 1, 2]))
                .then(pl.lit("Winter"))
                .when(pl.col("Month").is_in([3, 4, 5]))
                .then(pl.lit("Spring"))
                .when(pl.col("Month").is_in([6, 7, 8]))
                .then(pl.lit("Summer"))
                .when(pl.col("Month").is_in([9, 10, 11]))
                .then(pl.lit("Fall"))
                .otherwise(pl.lit("Unknown"))
                .alias("SEASON")
            ])
        
        # Add route (origin-dest)
        if "Origin" in df.columns and "Dest" in df.columns:
            df = df.with_columns([
                (pl.col("Origin") + "-" + pl.col("Dest")).alias("ROUTE")
            ])
        
        # Add delay categories
        if "ArrDelay" in df.columns:
            df = df.with_columns([
                pl.when(pl.col("ArrDelay").le(0))
                .then(pl.lit("On Time"))
                .when(pl.col("ArrDelay").le(15))
                .then(pl.lit("Minor Delay"))
                .when(pl.col("ArrDelay").le(60))
                .then(pl.lit("Moderate Delay"))
                .otherwise(pl.lit("Severe Delay"))
                .alias("DELAY_CATEGORY")
            ])
        
        return df
    
    def select_relevant_columns(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Select relevant columns for analysis"""
        self.logger.info("Selecting relevant columns")
        
        # Core columns for analysis - using actual CSV column names
        core_columns = [
            "Year", "Month", "DayofMonth", "DAY_OF_WEEK", "SEASON",
            "AIRLINE", "Origin", "Dest", "ROUTE",
            "CRSDepTime", "DEP_HOUR", "CRSArrTime",
            "DepDelay", "ArrDelay", "Cancelled", "Diverted",
            "Distance", "AirTime",
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
            
            # Simple approach: overwrite for now
            # In production, implement proper MERGE logic
            return new_data
            
        except Exception:
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
            partition_by=["Year", "Month"]
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
