
from pathlib import Path
import pyspark.sql.functions as F

dataset_root = Path.home() / "local" / "data"
yelp_dataset = dataset_root / "yelp" / "dataset" 

"""
dataset schema
 |-- business_id: string (nullable = true)
 |-- cool: long (nullable = true)
 |-- date: string (nullable = true)
 |-- funny: long (nullable = true)
 |-- review_id: string (nullable = true)
 |-- stars: long (nullable = true)
 |-- text: string (nullable = true)
 |-- useful: long (nullable = true)
 |-- user_id: string (nullable = true)
"""
df = spark.read.json(str(yelp_dataset / "review.json"))
df_resto_info = spark.read.json(str(yelp_dataset / "business.json"))

dates_distr = df.groupby("date").agg(F.count("date")).toPandas()

num_reviews_per_resto = df.groupby("business_id") \
                          .agg(F.count("review_id").alias("num_reviews")) \
                          .orderBy("num_reviews", ascending=False)

stars_distr = df.groupby("stars").agg(F.count("*")).toPandas()
stars_distr = stars_distr.sort_values("stars")

cities = df_resto_info.select("city", "postal_code", "state").distinct().toPandas()
calif_cities = cities['CA' == cities['state']]

calif_cities = cities[cities['postal_code'].apply(lambda s: s.startswith('94'))]

resto_review_join = num_reviews_per_resto.join(df_resto_info, "business_id")
