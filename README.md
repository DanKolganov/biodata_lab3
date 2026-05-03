# 🛩️ Flight Delay Analysis - Lakehouse Pipeline

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Polars](https://img.shields.io/badge/Polars-Lazy%20Evaluation-orange.svg)](https://pola.rs)
[![Delta Lake](https://img.shields.io/badge/Delta%20Lake-ACID%20Transactions-green.svg)](https://delta.io)
[![MLflow](https://img.shields.io/badge/MLflow-Model%20Tracking-blue.svg)](https://mlflow.org)

Complete lakehouse architecture implementation for US flight delays analysis (2018-2024) using **Polars** and **Delta Lake**.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Bronze Layer  │───▶│  Silver Layer   │───▶│   Gold Layer    │
│                 │    │                 │    │                 │
│ • Raw CSV data  │    │ • Cleaned data  │    │ • Analytics     │
│ • Versioning    │    │ • Enrichment     │    │ • Features      │
│ • Time travel   │    │ • Validation     │    │ • Aggregates    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │   ML Layer      │
                                              │                 │
                                              │ • Regression    │
                                              │ • Classification│
                                              │ • MLflow logging │
                                              └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (optional)
- Git

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd lab3
```

### 2. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs data/{bronze,silver,gold} mlflow
```

### 3. Prepare Data

```bash
# Copy your flight data CSV to data directory
cp your_flight_data.csv data/flight_data_2018_2024.csv
```

### 4. Run Pipeline

#### Option A: Individual Layers

```bash
# Run each layer separately
python -c "
import sys; sys.path.append('src')
from bronze.ingest import BronzeIngestion
from config import Config

bronze = BronzeIngestion()
bronze.ingest_by_years(Config.SOURCE_CSV, years=[2024])
print('✅ Bronze layer completed!')
"

python -c "
import sys; sys.path.append('src')
from silver.transform_fixed import SilverTransform

silver = SilverTransform()
silver.transform_pipeline()
print('✅ Silver layer completed!')
"

python -c "
import sys; sys.path.append('src')
from gold.simple_analytics import SimpleGoldAnalytics

gold = SimpleGoldAnalytics()
results = gold.create_all_analytics()
gold.show_summary(results)
print('✅ Gold layer completed!')
"

python -c "
import sys; sys.path.append('src')
from ml.simple_models import SimpleFlightML

ml = SimpleFlightML()
results = ml.run_ml_pipeline()
ml.show_results(results)
print('✅ ML layer completed!')
"
```

#### Option B: Docker Compose (Recommended)

```bash
# Build and run all services
docker-compose up --build

# Run full pipeline in container
docker-compose exec app python -m src.main --full
```

## 📊 Results

### Pipeline Performance
- **Bronze**: 582,425 records processed (2024 data)
- **Silver**: 556,894 cleaned records with enriched features
- **Gold**: 233 airports, 10 airlines, 145 temporal patterns
- **ML**: 10,000 feature records for model training

### Model Performance
| Model | Task | Metric | Score |
|-------|------|--------|-------|
| Linear Regression | Delay Prediction (min) | RMSE | 13.63 |
| Random Forest | Delay Prediction (min) | RMSE | 14.37 |
| Logistic Regression | Delay Classification (≥15min) | Accuracy | 92.5% |
| Random Forest | Delay Classification (≥15min) | Accuracy | 92.8% |

### Key Insights
- **Average delay**: 7.4 minutes
- **Delay rate**: 20.4% of flights
- **Worst airport**: CID (Cedar Rapids)
- **Best airport**: EWN (New Bern)
- **Worst airline**: AA (American Airlines)
- **Best airline**: WN (Southwest)

## 📁 Project Structure

```
lab3/
├── src/                          # Source code
│   ├── bronze/                   # Raw data ingestion
│   │   └── ingest.py
│   ├── silver/                   # Data transformation
│   │   └── transform_fixed.py
│   ├── gold/                     # Analytics & features
│   │   ├── aggregates.py
│   │   ├── aggregates_fixed.py
│   │   └── simple_analytics.py
│   ├── ml/                       # Machine learning
│   │   ├── models.py
│   │   └── simple_models.py
│   ├── config.py                 # Configuration
│   ├── delta_operations.py       # Delta Lake operations
│   └── main.py                   # Pipeline orchestrator
├── notebooks/                    # Jupyter notebooks
│   └── flight_delay_analysis.ipynb
├── data/                         # Data storage
│   ├── bronze/                   # Raw Delta tables
│   ├── silver/                   # Cleaned Delta tables
│   ├── gold/                     # Analytics & features
│   └── flight_data_2018_2024.csv  # Source data
├── logs/                         # Application logs
├── mlflow/                       # MLflow tracking
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container definition
├── docker-compose.yml           # Service orchestration
├── README.md                     # Detailed documentation
└── README_GITHUB.md             # This file
```

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
class Config:
    # Data paths
    SOURCE_CSV = DATA_DIR / "flight_data_2018_2024.csv"
    
    # ML settings
    DELAY_THRESHOLD_MINUTES = 15  # For classification
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Delta Lake settings
    DELTA_MAX_WORKERS = 4
    BATCH_SIZE = 10000
```

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t flight-delay-lakehouse .
```

### Run with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

### Access Services

- **MLflow UI**: http://localhost:5000
- **Application logs**: `docker-compose logs app`

## 🔍 Query Optimization

The pipeline uses Polars lazy evaluation for optimal performance:

```python
# Example optimized query
result = (pl.scan_delta("data/silver/flights_clean")
          .filter((pl.col("Year") == 2023) & 
                 (pl.col("AIRLINE").is_in(["AA", "DL", "UA"])))
          .group_by(["AIRLINE", "MONTH"])
          .agg([
              pl.len().alias("flight_count"),
              pl.col("ArrDelay").mean().alias("avg_delay")
          ])
          .sort("avg_delay", descending=True)
          .collect())
```

**Optimization Features:**
- ✅ Predicate pushdown
- ✅ Column pruning  
- ✅ Projection pushdown
- ✅ Partition pruning

## 📈 Delta Lake Features

### Time Travel

```python
# Read previous version
old_data = delta_ops.time_travel_query(table_path, version=5)

# Read as of timestamp
historical_data = delta_ops.time_travel_query(
    table_path, 
    timestamp="2024-01-15T10:00:00Z"
)
```

### Table Operations

```python
# Optimize table
delta_ops.optimize_table(table_path)

# Z-order by columns
delta_ops.z_order_table(table_path, ["Year", "Month", "AIRLINE"])

# Cleanup old files
delta_ops.vacuum_table(table_path, retention_hours=168)
```

## 🤖 Machine Learning

### Model Training

```python
from ml.simple_models import SimpleFlightML

ml = SimpleFlightML()
results = ml.run_ml_pipeline()

# Access trained models
reg_model = results['regression']['random_forest']['model']
clf_model = results['classification']['random_forest']['model']
```

### Feature Importance

```python
# Get top features
importance = results['feature_importance_regression']
for feature, score in list(importance.items())[:10]:
    print(f"{feature}: {score:.4f}")
```

## 📊 Analytics Examples

### Airport Performance

```python
from gold.simple_analytics import SimpleGoldAnalytics

gold = SimpleGoldAnalytics()
airport_stats = gold.create_airport_analytics(df)
print(airport_stats.sort("avg_arrival_delay", descending=True).head(10))
```

### Temporal Patterns

```python
temporal_stats = gold.create_temporal_analytics(df)
worst_hours = temporal_stats.sort("avg_arrival_delay", descending=True).head(5)
print(worst_hours.select(["DEP_HOUR", "avg_arrival_delay", "delay_rate"]))
```

## 🛠️ Troubleshooting

### Common Issues

1. **Memory Errors**
   ```bash
   # Reduce batch size in config.py
   BATCH_SIZE = 5000
   ```

2. **Delta Lake Compatibility**
   ```bash
   # Ensure data types are compatible
   # Check logs for schema errors
   ```

3. **MLflow Connection**
   ```bash
   # Start MLflow server first
   mlflow server --host 0.0.0.0 --port 5000
   ```

### Performance Tips

- Use lazy evaluation (`pl.scan_*` instead of `pl.read_*`)
- Filter data early in the pipeline
- Leverage partitioning for large datasets
- Use Z-ordering for frequently queried columns

## 📝 Logging

All operations are logged to `logs/` directory:

- `bronze_ingestion.log` - Raw data processing
- `silver_transform.log` - Data transformation
- `gold_aggregates.log` - Analytics creation
- `simple_ml.log` - Model training
- `delta_operations.log` - Delta Lake operations

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is for educational purposes as part of Big Data Tools coursework.

## 🙏 Acknowledgments

- [Polars](https://pola.rs/) - Fast DataFrame library
- [Delta Lake](https://delta.io/) - ACID transactions on data lakes
- [MLflow](https://mlflow.org/) - Machine learning lifecycle
- [US Bureau of Transportation Statistics](https://www.transtats.bts.gov/) - Flight data source

---

**🚀 Ready to analyze flight delays with lakehouse architecture!**
