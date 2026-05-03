import polars as pl
from deltalake import DeltaTable, write_deltalake
import logging
from typing import Dict

from config import Config

class GoldAggregates:
    """Gold layer: Analytical aggregates and feature tables"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.analytics_path = Config.GOLD_PATH / "analytics"
        self.features_path = Config.GOLD_PATH / "features"
        
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Config.LOGS_DIR / "gold_aggregates.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def load_silver_data(self) -> pl.LazyFrame:
        """Load data from silver layer"""
        self.logger.info("Loading data from silver layer")
        
        try:
            return pl.scan_delta(str(Config.SILVER_PATH / "flights_clean"))
        except Exception as e:
            self.logger.error(f"Error loading silver data: {str(e)}")
            # Fallback to CSV
            return pl.scan_csv(str(Config.SOURCE_CSV)).filter(pl.col("Year").eq(2024))
    
    def create_airport_analytics(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Create airport-level analytics"""
        self.logger.info("Creating airport analytics")
        
        airport_metrics = df.group_by(["Origin", "Year", "Month"]).agg([
            pl.len().alias("total_flights"),
            pl.col("ArrDelay").mean().alias("avg_arrival_delay"),
            pl.col("DepDelay").mean().alias("avg_departure_delay"),
            pl.col("ArrDelay").median().alias("median_arrival_delay"),
            pl.col("ArrDelay").quantile(0.95).alias("p95_arrival_delay"),
            (pl.col("ArrDelay") > 15).sum().alias("delayed_flights"),
            (pl.col("ArrDelay") > 15).mean().alias("delay_rate"),
            pl.col("Distance").mean().alias("avg_distance"),
            pl.col("AirTime").mean().alias("avg_air_time")
        ]).sort(["Year", "Month", "Origin"])
        
        return airport_metrics
    
    def create_airline_analytics(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Create airline-level analytics"""
        self.logger.info("Creating airline analytics")
        
        airline_metrics = df.group_by(["AIRLINE", "Year", "Month"]).agg([
            pl.len().alias("total_flights"),
            pl.col("ArrDelay").mean().alias("avg_arrival_delay"),
            pl.col("DepDelay").mean().alias("avg_departure_delay"),
            pl.col("ArrDelay").median().alias("median_arrival_delay"),
            (pl.col("ArrDelay") > 15).sum().alias("delayed_flights"),
            (pl.col("ArrDelay") > 15).mean().alias("delay_rate"),
            pl.col("Distance").mean().alias("avg_distance"),
            pl.col("AirTime").mean().alias("avg_air_time"),
            pl.col("ROUTE").n_unique().alias("unique_routes")
        ]).sort(["Year", "Month", "AIRLINE"])
        
        return airline_metrics
    
    def create_temporal_analytics(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Create temporal analytics (hour, day of week, season)"""
        self.logger.info("Creating temporal analytics")
        
        temporal_metrics = df.group_by(["Year", "Month", "DAY_OF_WEEK", "DEP_HOUR", "SEASON"]).agg([
            pl.len().alias("total_flights"),
            pl.col("ArrDelay").mean().alias("avg_arrival_delay"),
            pl.col("DepDelay").mean().alias("avg_departure_delay"),
            (pl.col("ArrDelay") > 15).sum().alias("delayed_flights"),
            (pl.col("ArrDelay") > 15).mean().alias("delay_rate"),
            pl.col("Distance").mean().alias("avg_distance")
        ]).sort(["Year", "Month", "DAY_OF_WEEK", "DEP_HOUR"])
        
        return temporal_metrics
    
    def create_route_analytics(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Create route-level analytics"""
        self.logger.info("Creating route analytics")
        
        route_metrics = df.group_by(["ROUTE", "Year", "Month"]).agg([
            pl.len().alias("total_flights"),
            pl.col("ArrDelay").mean().alias("avg_arrival_delay"),
            pl.col("DepDelay").mean().alias("avg_departure_delay"),
            (pl.col("ArrDelay") > 15).sum().alias("delayed_flights"),
            (pl.col("ArrDelay") > 15).mean().alias("delay_rate"),
            pl.col("Distance").first().alias("distance"),
            pl.col("AirTime").mean().alias("avg_air_time")
        ]).filter(pl.col("total_flights") >= 10).sort(["Year", "Month", "total_flights"])
        
        return route_metrics
    
    def create_feature_table(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Create feature table for ML modeling"""
        self.logger.info("Creating feature table")
        
        # Select features for ML
        feature_cols = [
            "Year", "Month", "DayofMonth", "DAY_OF_WEEK", "DEP_HOUR",
            "AIRLINE", "Origin", "Dest", "Distance", "AirTime",
            "DepDelay", "ArrDelay"
        ]
        
        # Filter available columns
        available_features = [col for col in feature_cols if col in df.columns]
        
        features_df = df.select(available_features).filter(
            pl.col("ArrDelay").is_not_null() &
            pl.col("DepDelay").is_not_null()
        )
        
        # Add target variables
        features_df = features_df.with_columns([
            pl.col("ArrDelay").alias("target_arrival_delay"),
            (pl.col("ArrDelay") > Config.DELAY_THRESHOLD_MINUTES).alias("target_is_delayed").cast(pl.Int32)
        ])
        
        return features_df
    
    def write_analytics_tables(self, analytics_dict: Dict[str, pl.LazyFrame]) -> None:
        """Write all analytics tables"""
        self.logger.info("Writing analytics tables")
        
        for table_name, df in analytics_dict.items():
            table_path = self.analytics_path / table_name
            
            try:
                # Convert to pandas
                pandas_df = df.collect().to_pandas()
                
                # Write with appropriate partitioning
                if table_name in ["airport_analytics", "airline_analytics"]:
                    partition_cols = ["Year", "Month"]
                elif table_name == "temporal_analytics":
                    partition_cols = ["Year", "Month"]
                else:
                    partition_cols = ["Year", "Month"]
                
                write_deltalake(
                    table_or_uri=str(table_path),
                    data=pandas_df,
                    mode="overwrite",
                    partition_by=partition_cols
                )
                
                self.logger.info(f"Written {table_name} with {len(pandas_df)} records")
                
            except Exception as e:
                self.logger.error(f"Error writing {table_name}: {str(e)}")
    
    def write_feature_table(self, df: pl.LazyFrame) -> None:
        """Write feature table"""
        self.logger.info("Writing feature table")
        
        try:
            pandas_df = df.collect().to_pandas()
            
            write_deltalake(
                table_or_uri=str(self.features_path),
                data=pandas_df,
                mode="overwrite",
                partition_by=["Year", "Month"]
            )
            
            self.logger.info(f"Written feature table with {len(pandas_df)} records")
            
        except Exception as e:
            self.logger.error(f"Error writing feature table: {str(e)}")
    
    def create_all_analytics(self) -> None:
        """Create all gold layer tables"""
        self.logger.info("Starting gold layer analytics creation")
        
        try:
            # Load silver data
            silver_df = self.load_silver_data()
            
            # Create all analytics
            airport_df = self.create_airport_analytics(silver_df)
            airline_df = self.create_airline_analytics(silver_df)
            temporal_df = self.create_temporal_analytics(silver_df)
            route_df = self.create_route_analytics(silver_df)
            features_df = self.create_feature_table(silver_df)
            
            # Write analytics tables
            analytics_dict = {
                "airport_analytics": airport_df,
                "airline_analytics": airline_df,
                "temporal_analytics": temporal_df,
                "route_analytics": route_df
            }
            
            self.write_analytics_tables(analytics_dict)
            self.write_feature_table(features_df)
            
            self.logger.info("Gold layer analytics creation completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in gold layer creation: {str(e)}")
            raise
    
    def get_table_stats(self) -> Dict[str, Dict]:
        """Get statistics for all gold tables"""
        stats = {}
        
        tables = {
            "airport_analytics": self.analytics_path / "airport_analytics",
            "airline_analytics": self.analytics_path / "airline_analytics", 
            "temporal_analytics": self.analytics_path / "temporal_analytics",
            "route_analytics": self.analytics_path / "route_analytics",
            "features": self.features_path
        }
        
        for table_name, table_path in tables.items():
            try:
                dt = DeltaTable(str(table_path))
                stats[table_name] = {
                    "version": dt.version(),
                    "files": len(dt.files()),
                    "rows": dt.to_pandas().shape[0]
                }
            except Exception as e:
                stats[table_name] = {"error": str(e)}
        
        return stats
