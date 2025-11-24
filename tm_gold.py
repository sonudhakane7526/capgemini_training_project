import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql import DataFrame

# 1. Setup
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'silver_bucket', 'gold_bucket'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

SILVER = f"s3://{args['silver_bucket']}"
GOLD = f"s3://{args['gold_bucket']}"
GOLD_DIM = f"{GOLD}/dim"
GOLD_FACT = f"{GOLD}/fact"
GOLD_RPT = f"{GOLD}/reports"

# ====================================================
# 0. UTILITY FUNCTIONS (Fixes Ambiguity)
# ====================================================
def smart_join(df1: DataFrame, df2: DataFrame, join_columns, join_type: str = "inner"):
    """
    Joins two DataFrames, keeps join keys, coalesces common columns, 
    and retains non-ambiguous columns from both sides.
    """
    a = df1.alias("a")
    b = df2.alias("b")
    
    joined = a.join(b, join_columns, join_type)

    # Identify columns
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    if isinstance(join_columns, str):
        join_columns = [join_columns]
    
    join_set = set(join_columns)
    common_cols = (cols1.intersection(cols2)) - join_set
    df1_only = cols1 - common_cols - join_set
    df2_only = cols2 - common_cols - join_set

    final_cols = []

    # 1. Add Join Columns
    for c in join_columns:
        final_cols.append(col("a." + c).alias(c))
    
    # 2. Coalesce Common Columns (Prioritize Left/A)
    for c in common_cols:
        final_cols.append(coalesce(col("a." + c), col("b." + c)).alias(c))
    
    # 3. Add Unique Columns from DF1
    for c in df1_only:
        final_cols.append(col("a." + c).alias(c))

    # 4. Add Unique Columns from DF2
    for c in df2_only:
        final_cols.append(col("b." + c).alias(c))
    
    return joined.select(*final_cols)

# ====================================================
# 2. Load Silver Data
# ====================================================
print("Loading Silver Data...")
players = spark.read.parquet(f"{SILVER}/players")
clubs = spark.read.parquet(f"{SILVER}/clubs")
competitions = spark.read.parquet(f"{SILVER}/competitions")
games = spark.read.parquet(f"{SILVER}/games")
appearances = spark.read.parquet(f"{SILVER}/appearances")
club_games = spark.read.parquet(f"{SILVER}/club_games")

# ====================================================
# 3. DIMENSION TABLES
# ====================================================
print("Creating Dimensions...")

# DIM_PLAYER
dim_player = players.select(
    col("player_id").alias("player_key"),
    "full_name", "first_name", "last_name", 
    "country_of_citizenship", "date_of_birth", "position", 
    "sub_position", "foot", "height_in_cm", 
    "current_club_name", "agent_name"
)
dim_player.write.mode("overwrite").parquet(f"{GOLD_DIM}/dim_player")

# DIM_CLUB
dim_club = clubs.select(
    col("club_id").alias("club_key"),
    "name", "club_code", "domestic_competition_id",
    "squad_size", "average_age", "foreigners_number",
    "stadium_name", "stadium_seats", "coach_name"
)
dim_club.write.mode("overwrite").parquet(f"{GOLD_DIM}/dim_club")

# DIM_COMPETITION
dim_competition = competitions.select(
    col("competition_id").alias("competition_key"),
    "name", "type", "country_name", "confederation", "is_international"
)
dim_competition.write.mode("overwrite").parquet(f"{GOLD_DIM}/dim_competition")

# DIM_DATE
distinct_dates = games.select("date").distinct().filter(col("date").isNotNull())
dim_date = distinct_dates \
    .withColumn("date_key", date_format(col("date"), "yyyyMMdd").cast("int")) \
    .withColumn("year", year(col("date"))) \
    .withColumn("month", month(col("date"))) \
    .withColumn("day", dayofmonth(col("date"))) \
    .withColumn("day_of_week", dayofweek(col("date"))) \
    .withColumn("day_name", date_format(col("date"), "EEEE")) \
    .withColumn("month_name", date_format(col("date"), "MMMM")) \
    .withColumn("is_weekend", when(col("day_of_week").isin(1, 7), lit(True)).otherwise(lit(False)))

dim_date.write.mode("overwrite").parquet(f"{GOLD_DIM}/dim_date")

# ====================================================
# 4. FACT TABLES (Using smart_join)
# ====================================================
print("Creating Facts...")

# FACT_PLAYER_APPEARANCE
# Using smart_join to handle ambiguous 'competition_id' or 'season' columns
fact_appearance = smart_join(appearances, games, "game_id") \
    .select(
        col("appearance_id").alias("appearance_key"),
        col("game_id").alias("game_key"),
        col("player_id").alias("player_key"),
        col("player_club_id").alias("club_key"),
        col("competition_id").alias("competition_key"), # Now safe due to smart_join
        date_format(col("date"), "yyyyMMdd").cast("int").alias("date_key"),
        col("season_year"),
        "goals", "assists", "yellow_cards", "red_cards", "minutes_played"
    )
fact_appearance.write.mode("overwrite").parquet(f"{GOLD_FACT}/fact_player_appearance")

# FACT_GAME
fact_game = games.select(
    col("game_id").alias("game_key"),
    col("competition_id").alias("competition_key"),
    col("home_club_id").alias("home_club_key"),
    col("away_club_id").alias("away_club_key"),
    date_format(col("date"), "yyyyMMdd").cast("int").alias("date_key"),
    col("season_year"),
    "round",
    "home_club_goals", "away_club_goals", "total_goals",
    "attendance"
)
fact_game.write.mode("overwrite").parquet(f"{GOLD_FACT}/fact_game")

# FACT_CLUB_PERFORMANCE
fact_club_perf = club_games.select(
    col("game_id").alias("game_key"),
    col("club_id").alias("club_key"),
    col("opponent_id").alias("opponent_club_key"),
    "own_goals", "opponent_goals", "hosting", "is_win"
)
fact_club_perf.write.mode("overwrite").parquet(f"{GOLD_FACT}/fact_club_performance")

# ====================================================
# 5. BUSINESS REPORTS
# ====================================================
print("Generating Business Reports...")

# 1. Player Country & Clubs
player_clubs = fact_appearance.join(dim_club, "club_key") \
    .groupBy("player_key").agg(countDistinct("name").alias("number_of_clubs"))

report_1 = dim_player.join(player_clubs, "player_key") \
    .select("full_name", "country_of_citizenship", "position", "number_of_clubs") \
    .orderBy(desc("number_of_clubs"))
report_1.write.mode("overwrite").parquet(f"{GOLD_RPT}/report_player_country_clubs")

# 2. Top International Appearances
intl_stats = fact_appearance.join(dim_competition, "competition_key") \
    .filter(col("is_international") == True) \
    .groupBy("player_key") \
    .agg(count("*").alias("intl_apps"), sum("goals").alias("intl_goals"))

report_2 = dim_player.join(intl_stats, "player_key") \
    .select("full_name", "country_of_citizenship", "intl_apps", "intl_goals") \
    .orderBy(desc("intl_apps"))
report_2.write.mode("overwrite").parquet(f"{GOLD_RPT}/report_international_appearances")

# 3. Top Scorers 2014
scorers_2014 = fact_appearance.filter(col("season_year") == 2014) \
    .groupBy("player_key").agg(sum("goals").alias("total_goals"))

report_3 = dim_player.join(scorers_2014, "player_key") \
    .select("full_name", "total_goals", "country_of_citizenship") \
    .orderBy(desc("total_goals")) \
    .limit(10)
report_3.write.mode("overwrite").parquet(f"{GOLD_RPT}/report_top_scorers_2014")

# 4. Top Red Cards
red_cards = fact_appearance.groupBy("player_key") \
    .agg(sum("red_cards").alias("total_reds"), count("game_key").alias("total_games")) \
    .filter(col("total_reds") > 0)

report_4 = dim_player.join(red_cards, "player_key") \
    .select("full_name", "total_reds", "total_games") \
    .orderBy(desc("total_reds"))
report_4.write.mode("overwrite").parquet(f"{GOLD_RPT}/report_red_cards")

# 5. Successful Clubs
club_wins = fact_club_perf.groupBy("club_key") \
    .agg(sum(when(col("is_win") == "win", 1).otherwise(0)).alias("total_wins"))

report_5 = dim_club.join(club_wins, "club_key") \
    .select("name", "total_wins") \
    .orderBy(desc("total_wins")) \
    .limit(20)
report_5.write.mode("overwrite").parquet(f"{GOLD_RPT}/report_successful_clubs")

# 6. Player Performance Summary
report_6 = fact_appearance.groupBy("player_key") \
    .agg(
        sum("goals").alias("total_goals"),
        sum("assists").alias("total_assists"),
        count("game_key").alias("apps")
    ) \
    .join(dim_player, "player_key") \
    .select("full_name", "total_goals", "total_assists", "apps") \
    .orderBy(desc("total_goals"))
report_6.write.mode("overwrite").parquet(f"{GOLD_RPT}/report_player_performance")

# 7. Competition Statistics
report_7 = fact_game.join(dim_competition, "competition_key") \
    .groupBy("name", "type", "country_name") \
    .agg(
        count("game_key").alias("total_games"),
        avg("total_goals").alias("avg_goals"),
        avg("attendance").alias("avg_attendance")
    ) \
    .orderBy(desc("avg_attendance"))
report_7.write.mode("overwrite").parquet(f"{GOLD_RPT}/report_competition_stats")

# 8. Top Assist Providers
report_8 = fact_appearance.groupBy("player_key") \
    .agg(sum("assists").alias("total_assists")) \
    .join(dim_player, "player_key") \
    .select("full_name", "total_assists") \
    .orderBy(desc("total_assists")) \
    .limit(20)
report_8.write.mode("overwrite").parquet(f"{GOLD_RPT}/report_top_assists")

# 9. Home vs Away Performance
report_9 = fact_club_perf.groupBy("club_key") \
    .agg(
        sum(when((col("hosting") == "Home") & (col("is_win") == "win"), 1).otherwise(0)).alias("home_wins"),
        sum(when((col("hosting") == "Away") & (col("is_win") == "win"), 1).otherwise(0)).alias("away_wins")
    ) \
    .join(dim_club, "club_key") \
    .select("name", "home_wins", "away_wins")
report_9.write.mode("overwrite").parquet(f"{GOLD_RPT}/report_home_away_performance")

# 10. Season Trends
report_10 = fact_game.groupBy("season_year") \
    .agg(
        sum("total_goals").alias("total_goals"),
        avg("attendance").alias("avg_attendance")
    ) \
    .orderBy("season_year")
report_10.write.mode("overwrite").parquet(f"{GOLD_RPT}/report_season_trends")

# 11. Position Analysis
report_11 = fact_appearance.join(dim_player, "player_key") \
    .groupBy("position") \
    .agg(
        avg("goals").alias("avg_goals_per_player"),
        avg("assists").alias("avg_assists_per_player")
    ) \
    .orderBy(desc("avg_goals_per_player"))
report_11.write.mode("overwrite").parquet(f"{GOLD_RPT}/report_position_analysis")

# 12. Top Derby Matches (High Attendance)
home_c = dim_club.select(col("club_key").alias("home_club_key"), col("name").alias("home_name"))
away_c = dim_club.select(col("club_key").alias("away_club_key"), col("name").alias("away_name"))

report_12 = fact_game \
    .join(home_c, "home_club_key") \
    .join(away_c, "away_club_key") \
    .select("home_name", "away_name", "attendance", "total_goals", "date_key") \
    .orderBy(desc("attendance")) \
    .limit(50)
report_12.write.mode("overwrite").parquet(f"{GOLD_RPT}/report_top_matches")

# 13. Nationality Distribution
report_13 = dim_player.groupBy("country_of_citizenship") \
    .count().withColumnRenamed("count", "player_count") \
    .orderBy(desc("player_count"))
report_13.write.mode("overwrite").parquet(f"{GOLD_RPT}/report_nationality_distribution")

# 14. Monthly Trends
report_14 = fact_game.join(dim_date, "date_key") \
    .groupBy("month_name", "month") \
    .agg(sum("total_goals").alias("goals")) \
    .orderBy("month")
report_14.write.mode("overwrite").parquet(f"{GOLD_RPT}/report_monthly_trends")

# 15. Squad Analysis
report_15 = dim_club.select("name", "squad_size", "foreigners_number", "average_age") \
    .orderBy(desc("squad_size"))
report_15.write.mode("overwrite").parquet(f"{GOLD_RPT}/report_squad_analysis")

print("JOB SUCCESS: Gold Layer & 15 Reports Generated")