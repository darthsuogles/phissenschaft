import pyspark.sql.functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType

"""Please check Spark documentation for SQL functions
https://spark.apache.org/docs/2.3.2/api/python/pyspark.sql.html#module-pyspark.sql.functions
"""
df = spark.range(10000)
df.printSchema()
df.select(F.sum('id')).collect()
df_quantiles = df.approxQuantile(
    'id', [0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 0.99], 0.3)

# To rum distributed inference with GPU, the best way is still to
# launch a separate Horovod job and get the result stored in
# a mutually recognizatble format (e.g. TFRecord or Parquet)
