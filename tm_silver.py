import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from pyspark.sql.functions import *

# 1. Setup Arguments & Context
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'bronze_bucket', 'silver_bucket'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

BRONZE = f"s3://{args['bronze_bucket']}"
SILVER = f"s3://{args['silver_bucket']}"

# 2. Helper Function: Convert "null"/"NULL" strings to actual SQL Nulls
def clean_strings(df):
    for name, dtype in df.dtypes:
        if dtype == "string":
            df = df.withColumn(name, when(col(name).isin("null", "NULL", "Null", ""), None).otherwise(col(name)))
    return df

# ==========================================
# 3. Transformation Logic (All 8 Datasets)
# ==========================================

# --- 1. PLAYERS ---
print("Processing Players...")
df_players = spark.read.parquet(f"{BRONZE}/players")
df_players = clean_strings(df_players) \
    .dropDuplicates(["player_id"]) \
    .na.fill({
        "first_name": "Unknown",
        "last_name": "Unknown",
        "country_of_citizenship": "Unknown",
        "position": "Unknown",
        "sub_position": "Unknown"
    }) \
    .withColumn("full_name", 
               when(col("name").isNotNull(), col("name"))
               .otherwise(concat_ws(" ", col("first_name"), col("last_name")))) \
    .withColumn("date_of_birth", to_date(col("date_of_birth")))

df_players.write.mode("overwrite").parquet(f"{SILVER}/players")


# --- 2. APPEARANCES ---
print("Processing Appearances...")
df_apps = spark.read.parquet(f"{BRONZE}/appearances")
df_apps = clean_strings(df_apps) \
    .filter(col("player_id").isNotNull()) \
    .filter(col("game_id").isNotNull()) \
    .na.fill({
        "yellow_cards": 0, 
        "red_cards": 0, 
        "goals": 0, 
        "assists": 0, 
        "minutes_played": 0
    }) \
    .withColumn("date", to_date(col("date")))

df_apps.write.mode("overwrite").parquet(f"{SILVER}/appearances")


# --- 3. GAMES ---
print("Processing Games...")
df_games = spark.read.parquet(f"{BRONZE}/games")
df_games = clean_strings(df_games) \
    .dropDuplicates(["game_id"]) \
    .filter(col("game_id").isNotNull()) \
    .withColumn("date", to_date(col("date"))) \
    .withColumn("season_year", col("season").cast("int")) \
    .withColumn("total_goals", col("home_club_goals") + col("away_club_goals"))

df_games.write.mode("overwrite").parquet(f"{SILVER}/games")


# --- 4. CLUBS ---
print("Processing Clubs...")
df_clubs = spark.read.parquet(f"{BRONZE}/clubs")
df_clubs = clean_strings(df_clubs) \
    .dropDuplicates(["club_id"]) \
    .na.fill({
        "name": "Unknown Club",
        "squad_size": 0,
        "average_age": 0.0,
        "foreigners_number": 0
    })

df_clubs.write.mode("overwrite").parquet(f"{SILVER}/clubs")


# --- 5. COMPETITIONS ---
print("Processing Competitions...")
df_comps = spark.read.parquet(f"{BRONZE}/competitions")
df_comps = clean_strings(df_comps) \
    .dropDuplicates(["competition_id"]) \
    .na.fill({"country_name": "Unknown"}) \
    .withColumn("is_international", 
               when(col("type").isin("international_cup", "international"), lit(True))
               .otherwise(lit(False)))

df_comps.write.mode("overwrite").parquet(f"{SILVER}/competitions")


# --- 6. GAME EVENTS ---
print("Processing Game Events...")
df_events = spark.read.parquet(f"{BRONZE}/game_events")
df_events = clean_strings(df_events) \
    .filter(col("game_id").isNotNull()) \
    .na.fill({"description": ""})

df_events.write.mode("overwrite").parquet(f"{SILVER}/game_events")


# --- 7. CLUB GAMES ---
print("Processing Club Games...")
df_club_games = spark.read.parquet(f"{BRONZE}/club_games")
df_club_games = clean_strings(df_club_games) \
    .dropDuplicates(["game_id", "club_id"])

df_club_games.write.mode("overwrite").parquet(f"{SILVER}/club_games")


# --- 8. PLAYER VALUATIONS ---
print("Processing Player Valuations...")
df_vals = spark.read.parquet(f"{BRONZE}/player_valuations")
df_vals = clean_strings(df_vals) \
    .withColumn("date", to_date(col("date"))) \
    .withColumn("datetime", to_date(col("datetime")))

df_vals.write.mode("overwrite").parquet(f"{SILVER}/player_valuations")

print("JOB SUCCESS: Silver Layer Complete (8 Tables)")
