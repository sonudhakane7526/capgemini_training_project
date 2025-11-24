import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from pyspark.sql import SparkSession

# Get arguments passed from Airflow
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'raw_bucket', 'bronze_bucket'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

RAW_BUCKET = args['raw_bucket']
BRONZE_BUCKET = args['bronze_bucket']

files = [
    "players", "appearances", "club_games", "clubs", 
    "competitions", "game_events", "games", "player_valuations"
]

for file in files:
    source_path = f"s3://{RAW_BUCKET}/data/{file}.csv"
    target_path = f"s3://{BRONZE_BUCKET}/{file}"
    
    print(f"Ingesting {source_path} -> {target_path}")
    
    try:
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(source_path)
        df.write.mode("overwrite").parquet(target_path)
    except Exception as e:
        print(f"Skipping {file}: {str(e)}")

print("JOB SUCCESS: Bronze Layer Created")
