import polars as pl
from deltalake import DeltaTable, write_deltalake
from pathlib import Path
import logging
from datetime import datetime
from typing import Optional

from config import Config

class BronzeIngestion:
    """Bronze layer: Raw data ingestion with Delta Lake versioning"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.bronze_path = Config.BRONZE_PATH / "flights"
        
    def _setup_logger(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Config.LOGS_DIR / "bronze_ingestion.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def load_source_data(self, file_path: Path) -> pl.LazyFrame:
        """Load source CSV data with lazy evaluation"""
        self.logger.info(f"Loading source data from {file_path}")
        
        # Use lazy scan for memory efficiency
        df = pl.scan_csv(file_path)
        
        # Add metadata
        df = df.with_columns([
            pl.lit(datetime.now()).alias("ingested_at"),
            pl.lit("bronze").alias("source_layer")
        ])
        
        return df
    
    def filter_by_year(self, df: pl.LazyFrame, year: int) -> pl.LazyFrame:
        """Filter data by specific year for batch processing"""
        return df.filter(pl.col("Year").eq(year))
    
    def write_to_delta(self, df: pl.LazyFrame, year: int, mode: str = "append") -> None:
        """Write data to Delta Lake table"""
        self.logger.info(f"Writing {year} data to Delta Lake with mode: {mode}")
        
        # Sample a small batch to determine column types
        sample_df = df.limit(100).collect()
        
        # Handle null values based on column types
        fill_expressions = []
        for col_name in sample_df.columns:
            dtype = sample_df[col_name].dtype
            
            if dtype in [pl.Float64, pl.Float32]:
                fill_expressions.append(pl.col(col_name).fill_null(0.0))
            elif dtype in [pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
                fill_expressions.append(pl.col(col_name).fill_null(0))
            elif dtype == pl.String:
                fill_expressions.append(pl.col(col_name).fill_null(""))
            elif dtype == pl.Boolean:
                fill_expressions.append(pl.col(col_name).fill_null(False))
            else:
                # For other types, try to infer appropriate fill value
                try:
                    fill_expressions.append(pl.col(col_name).fill_null(''))
                except:
                    fill_expressions.append(pl.col(col_name).fill_null(0))
        
        if fill_expressions:
            df = df.with_columns(fill_expressions)
        
        # Collect to pandas for Delta Lake compatibility
        pandas_df = df.collect().to_pandas()
        
        # Additional null handling in pandas
        pandas_df = pandas_df.fillna('')
        
        # Write to Delta Lake
        write_deltalake(
            table_or_uri=str(self.bronze_path),
            data=pandas_df,
            mode=mode,
            partition_by=["Year"]
        )
        
        self.logger.info(f"Successfully wrote {len(pandas_df)} records for year {year}")
    
    def ingest_by_years(self, file_path: Path, years: Optional[list] = None) -> None:
        """Ingest data by years with incremental loading"""
        self.logger.info("Starting bronze layer ingestion by years")
        
        # Default years from dataset
        if years is None:
            years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
        
        # Load source data once
        source_df = self.load_source_data(file_path)
        
        # Process each year
        for year in years:
            try:
                self.logger.info(f"Processing year {year}")
                
                # Filter by year
                year_df = self.filter_by_year(source_df, year)
                
                # Check if we have data for this year
                sample_count = year_df.select(pl.len()).collect().item()
                if sample_count == 0:
                    self.logger.warning(f"No data found for year {year}")
                    continue
                
                # Write to Delta Lake
                self.write_to_delta(year_df, year, mode="append")
                
            except Exception as e:
                self.logger.error(f"Error processing year {year}: {str(e)}")
                continue
        
        self.logger.info("Bronze layer ingestion completed")
    
    def get_table_info(self) -> dict:
        """Get Delta table information"""
        try:
            dt = DeltaTable(str(self.bronze_path))
            return {
                "version": dt.version(),
                "files": dt.file_uris(),
                "metadata": dt.metadata(),
                "partition_columns": dt.metadata().partition_columns
            }
        except Exception as e:
            self.logger.error(f"Error getting table info: {str(e)}")
            return {}
    
    def time_travel_read(self, version: int) -> pl.DataFrame:
        """Read data from specific version (time travel)"""
        try:
            dt = DeltaTable(str(self.bronze_path))
            df = pl.from_pandas(dt.to_pandas(version=version))
            self.logger.info(f"Read data from version {version}")
            return df
        except Exception as e:
            self.logger.error(f"Error reading version {version}: {str(e)}")
            return pl.DataFrame()
