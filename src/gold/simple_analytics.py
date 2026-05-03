import polars as pl
import logging
from pathlib import Path

from config import Config

class SimpleGoldAnalytics:
    """Simplified Gold layer for demonstration"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.analytics_path = Config.GOLD_PATH / "analytics"
        self.features_path = Config.GOLD_PATH / "features"
        
        # Ensure directories exist
        self.analytics_path.mkdir(parents=True, exist_ok=True)
        self.features_path.mkdir(parents=True, exist_ok=True)
        
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Config.LOGS_DIR / "simple_gold.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def load_data(self) -> pl.LazyFrame:
        """Load data from CSV"""
        self.logger.info("Loading data from CSV")
        return pl.scan_csv(str(Config.SOURCE_CSV)).filter(pl.col("Year").eq(2024))
    
    def create_airport_analytics(self, df: pl.LazyFrame) -> pl.DataFrame:
        """Create airport-level analytics"""
        self.logger.info("Creating airport analytics")
        
        return (df.group_by(["Origin", "Month"])
                .agg([
                    pl.len().alias("total_flights"),
                    pl.col("ArrDelay").mean().alias("avg_arrival_delay"),
                    pl.col("DepDelay").mean().alias("avg_departure_delay"),
                    pl.col("ArrDelay").median().alias("median_arrival_delay"),
                    (pl.col("ArrDelay") > 15).sum().alias("delayed_flights"),
                    (pl.col("ArrDelay") > 15).mean().alias("delay_rate"),
                    pl.col("Distance").mean().alias("avg_distance")
                ])
                .filter(pl.col("total_flights") >= 100)
                .sort(["Month", "total_flights"], descending=[False, True])
                .collect())
    
    def create_airline_analytics(self, df: pl.LazyFrame) -> pl.DataFrame:
        """Create airline-level analytics"""
        self.logger.info("Creating airline analytics")
        
        return (df.group_by(["IATA_Code_Marketing_Airline", "Month"])
                .agg([
                    pl.len().alias("total_flights"),
                    pl.col("ArrDelay").mean().alias("avg_arrival_delay"),
                    pl.col("DepDelay").mean().alias("avg_departure_delay"),
                    (pl.col("ArrDelay") > 15).sum().alias("delayed_flights"),
                    (pl.col("ArrDelay") > 15).mean().alias("delay_rate"),
                    pl.col("Distance").mean().alias("avg_distance")
                ])
                .filter(pl.col("total_flights") >= 1000)
                .sort(["Month", "total_flights"], descending=[False, True])
                .collect())
    
    def create_temporal_analytics(self, df: pl.LazyFrame) -> pl.DataFrame:
        """Create temporal analytics"""
        self.logger.info("Creating temporal analytics")
        
        # Add hour column
        df_with_hour = df.with_columns([
            (pl.col("CRSDepTime") // 100).alias("DEP_HOUR").cast(pl.Int32)
        ])
        
        return (df_with_hour.group_by(["Month", "DayOfWeek", "DEP_HOUR"])
                .agg([
                    pl.len().alias("total_flights"),
                    pl.col("ArrDelay").mean().alias("avg_arrival_delay"),
                    (pl.col("ArrDelay") > 15).sum().alias("delayed_flights"),
                    (pl.col("ArrDelay") > 15).mean().alias("delay_rate")
                ])
                .filter(pl.col("total_flights") >= 50)
                .sort(["Month", "DayOfWeek", "DEP_HOUR"])
                .collect())
    
    def create_feature_table(self, df: pl.LazyFrame) -> pl.DataFrame:
        """Create feature table for ML"""
        self.logger.info("Creating feature table")
        
        # Add derived features
        df_features = df.with_columns([
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
        
        return (df_features.select(feature_cols)
                .filter(
                    pl.col("ArrDelay").is_not_null() &
                    pl.col("DepDelay").is_not_null() &
                    pl.col("Distance").gt(0)
                )
                .with_columns([
                    pl.col("ArrDelay").alias("target_arrival_delay"),
                    (pl.col("ArrDelay") > 15).alias("target_is_delayed").cast(pl.Int32)
                ])
                .limit(10000)  # Sample for demo
                .collect())
    
    def save_to_csv(self, df: pl.DataFrame, filename: str) -> None:
        """Save DataFrame to CSV"""
        filepath = self.analytics_path / filename
        df.write_csv(str(filepath))
        self.logger.info(f"Saved {filename} with {len(df)} records")
    
    def create_all_analytics(self) -> dict:
        """Create all analytics and return results"""
        self.logger.info("Starting simple gold analytics")
        
        # Load data
        df = self.load_data()
        
        # Create analytics
        airport_df = self.create_airport_analytics(df)
        airline_df = self.create_airline_analytics(df)
        temporal_df = self.create_temporal_analytics(df)
        features_df = self.create_feature_table(df)
        
        # Save results
        self.save_to_csv(airport_df, "airport_analytics.csv")
        self.save_to_csv(airline_df, "airline_analytics.csv")
        self.save_to_csv(temporal_df, "temporal_analytics.csv")
        features_df.write_csv(str(self.features_path / "features.csv"))
        
        self.logger.info(f"Saved features table with {len(features_df)} records")
        
        return {
            "airport_analytics": airport_df,
            "airline_analytics": airline_df,
            "temporal_analytics": temporal_df,
            "features": features_df
        }
    
    def show_summary(self, results: dict) -> None:
        """Show summary of created analytics"""
        print("\n" + "="*60)
        print("GOLD LAYER ANALYTICS SUMMARY")
        print("="*60)
        
        for name, df in results.items():
            print(f"\n{name.upper()}:")
            print(f"  Records: {len(df):,}")
            print(f"  Columns: {len(df.columns)}")
            
            if name == "airport_analytics":
                worst_airport = df.sort("avg_arrival_delay", descending=True).row(0)[0]
                best_airport = df.sort("avg_arrival_delay").row(0)[0]
                print(f"  Worst delay airport: {worst_airport}")
                print(f"  Best delay airport: {best_airport}")
                
            elif name == "airline_analytics":
                worst_airline = df.sort("avg_arrival_delay", descending=True).row(0)[0]
                best_airline = df.sort("avg_arrival_delay").row(0)[0]
                print(f"  Worst delay airline: {worst_airline}")
                print(f"  Best delay airline: {best_airline}")
                
            elif name == "features":
                avg_delay = df["target_arrival_delay"].mean()
                delay_rate = df["target_is_delayed"].mean()
                print(f"  Average delay: {avg_delay:.1f} minutes")
                print(f"  Delay rate: {delay_rate:.1%}")
        
        print("\n" + "="*60)
