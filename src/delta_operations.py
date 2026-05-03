import polars as pl
from deltalake import DeltaTable
import logging
from pathlib import Path
from typing import List, Dict, Optional

from config import Config

class DeltaOperations:
    """Delta Lake advanced operations: OPTIMIZE, Z-ORDER, VACUUM, time travel"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Config.LOGS_DIR / "delta_operations.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def optimize_table(self, table_path: str, target_size: int = 128 * 1024 * 1024) -> None:
        """OPTIMIZE operation for file compaction"""
        self.logger.info(f"Optimizing table: {table_path}")
        
        try:
            dt = DeltaTable(table_path)
            
            # Perform optimize
            dt.optimize(
                target_size=target_size,
                max_concurrent_tasks=Config.DELTA_MAX_WORKERS
            )
            
            self.logger.info(f"Table optimization completed: {table_path}")
            
        except Exception as e:
            self.logger.error(f"Error optimizing table {table_path}: {str(e)}")
            raise
    
    def z_order_table(self, table_path: str, columns: List[str]) -> None:
        """Z-ORDER operation for clustering data"""
        self.logger.info(f"Z-ordering table: {table_path} on columns: {columns}")
        
        try:
            dt = DeltaTable(table_path)
            
            # Perform z-order
            dt.z_order(columns, max_concurrent_tasks=Config.DELTA_MAX_WORKERS)
            
            self.logger.info(f"Z-ordering completed: {table_path}")
            
        except Exception as e:
            self.logger.error(f"Error z-ordering table {table_path}: {str(e)}")
            raise
    
    def vacuum_table(self, table_path: str, retention_hours: int = 168) -> int:
        """VACUUM operation to remove old files"""
        self.logger.info(f"Vacuuming table: {table_path} with retention {retention_hours}h")
        
        try:
            dt = DeltaTable(table_path)
            
            # Perform vacuum (dry run first for safety)
            dry_run_files = dt.vacuum(retention_hours=retention_hours, dry_run=True)
            self.logger.info(f"VACUUM dry run: {len(dry_run_files)} files would be deleted")
            
            # Actual vacuum
            deleted_files = dt.vacuum(retention_hours=retention_hours, dry_run=False)
            
            self.logger.info(f"VACUUM completed: {len(deleted_files)} files deleted")
            return len(deleted_files)
            
        except Exception as e:
            self.logger.error(f"Error vacuuming table {table_path}: {str(e)}")
            raise
    
    def time_travel_query(self, table_path: str, version: Optional[int] = None, 
                         timestamp: Optional[str] = None) -> pl.DataFrame:
        """Read data from specific version or timestamp (time travel)"""
        if version:
            self.logger.info(f"Reading version {version} from table: {table_path}")
            version_info = f"version={version}"
        elif timestamp:
            self.logger.info(f"Reading data as of {timestamp} from table: {table_path}")
            version_info = f"timestamp={timestamp}"
        else:
            raise ValueError("Either version or timestamp must be provided")
        
        try:
            dt = DeltaTable(table_path)
            
            if version:
                pandas_df = dt.to_pandas(version=version)
            else:
                pandas_df = dt.to_pandas(timestamp=timestamp)
            
            polars_df = pl.from_pandas(pandas_df)
            
            self.logger.info(f"Time travel query completed: {version_info}, rows: {len(polars_df)}")
            return polars_df
            
        except Exception as e:
            self.logger.error(f"Error in time travel query: {str(e)}")
            raise
    
    def get_table_history(self, table_path: str) -> List[Dict]:
        """Get table version history"""
        self.logger.info(f"Getting history for table: {table_path}")
        
        try:
            dt = DeltaTable(table_path)
            history = dt.history()
            
            self.logger.info(f"Table history retrieved: {len(history)} versions")
            return history
            
        except Exception as e:
            self.logger.error(f"Error getting table history: {str(e)}")
            raise
    
    def get_table_stats(self, table_path: str) -> Dict:
        """Get detailed table statistics"""
        self.logger.info(f"Getting stats for table: {table_path}")
        
        try:
            dt = DeltaTable(table_path)
            
            # Get basic info
            metadata = dt.metadata()
            version = dt.version()
            files = dt.files()
            
            # Get file sizes
            total_size = 0
            file_count = len(files)
            
            stats = {
                "version": version,
                "file_count": file_count,
                "partition_columns": metadata.partition_columns,
                "configuration": metadata.configuration,
                "created_at": metadata.created_at,
                "files": files[:5]  # First 5 files for preview
            }
            
            self.logger.info(f"Table stats retrieved: {table_path}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting table stats: {str(e)}")
            raise
    
    def schema_evolution_demo(self, table_path: str) -> None:
        """Demonstrate schema evolution capabilities"""
        self.logger.info("Demonstrating schema evolution")
        
        try:
            dt = DeltaTable(table_path)
            
            # Get current schema
            current_schema = dt.schema().to_pyarrow()
            self.logger.info(f"Current schema: {current_schema}")
            
            # Schema evolution would happen automatically when writing
            # with additional columns - this is just a demonstration
            
            self.logger.info("Schema evolution demo completed")
            
        except Exception as e:
            self.logger.error(f"Error in schema evolution demo: {str(e)}")
            raise
    
    def optimize_all_tables(self) -> None:
        """Optimize all Delta tables in the lakehouse"""
        self.logger.info("Starting optimization of all tables")
        
        tables = [
            str(Config.BRONZE_PATH / "flights"),
            str(Config.SILVER_PATH / "flights_clean"),
            str(Config.GOLD_PATH / "analytics/airport_analytics"),
            str(Config.GOLD_PATH / "analytics/airline_analytics"),
            str(Config.GOLD_PATH / "analytics/temporal_analytics"),
            str(Config.GOLD_PATH / "analytics/route_analytics"),
            str(Config.GOLD_PATH / "features")
        ]
        
        for table_path in tables:
            try:
                if Path(table_path).exists():
                    self.optimize_table(table_path)
                    
                    # Z-order key columns for each table
                    if "bronze" in table_path:
                        self.z_order_table(table_path, ["YEAR"])
                    elif "silver" in table_path:
                        self.z_order_table(table_path, ["YEAR", "MONTH", "AIRLINE"])
                    elif "features" in table_path:
                        self.z_order_table(table_path, ["YEAR", "MONTH", "AIRLINE"])
                    else:
                        self.z_order_table(table_path, ["YEAR", "MONTH"])
                        
            except Exception as e:
                self.logger.error(f"Error optimizing table {table_path}: {str(e)}")
                continue
        
        self.logger.info("All tables optimization completed")
    
    def cleanup_old_data(self) -> None:
        """Clean up old data with VACUUM"""
        self.logger.info("Starting cleanup of old data")
        
        tables = [
            str(Config.BRONZE_PATH / "flights"),
            str(Config.SILVER_PATH / "flights_clean"),
            str(Config.GOLD_PATH / "features")
        ]
        
        for table_path in tables:
            try:
                if Path(table_path).exists():
                    deleted_count = self.vacuum_table(table_path, retention_hours=168)  # 7 days
                    self.logger.info(f"Cleaned {deleted_count} files from {table_path}")
            except Exception as e:
                self.logger.error(f"Error cleaning table {table_path}: {str(e)}")
                continue
        
        self.logger.info("Data cleanup completed")
    
    def demonstrate_time_travel(self) -> Dict[str, pl.DataFrame]:
        """Demonstrate time travel capabilities"""
        self.logger.info("Demonstrating time travel")
        
        results = {}
        
        try:
            # Get history for silver table
            silver_path = str(Config.SILVER_PATH / "flights_clean")
            if Path(silver_path).exists():
                history = self.get_table_history(silver_path)
                
                if len(history) >= 2:
                    # Get latest and previous version
                    latest_version = history[0]["version"]
                    previous_version = history[1]["version"] if len(history) > 1 else latest_version - 1
                    
                    if previous_version >= 0:
                        results["latest"] = self.time_travel_query(silver_path, version=latest_version)
                        results["previous"] = self.time_travel_query(silver_path, version=previous_version)
                        
                        self.logger.info(f"Time travel demo: latest={len(results['latest'])}, previous={len(results['previous'])}")
                
        except Exception as e:
            self.logger.error(f"Error in time travel demo: {str(e)}")
        
        return results
