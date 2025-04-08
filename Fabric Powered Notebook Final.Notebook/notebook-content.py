# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "486921fd-3eb3-47c0-b9ce-f7804a355198",
# META       "default_lakehouse_name": "energy_lakehouse",
# META       "default_lakehouse_workspace_id": "1cf87aea-aa0c-4a93-80a1-929e61be15e5",
# META       "known_lakehouses": [
# META         {
# META           "id": "486921fd-3eb3-47c0-b9ce-f7804a355198"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# #### **Fabric Powered Energy Horizon**

# MARKDOWN ********************

# Welcome to our project submission for the Microsoft Hackathon 2025. This notebook presents a data-driven approach to forecasting energy generation trends using historical datasets on electricity production. The primary objective is to develop a simple but effective machine learning model that can predict future energy trends — especially from renewable sources — across different regions.
# 
# This solution is built with scalability and interpretability in mind and is designed to integrate with Microsoft Fabric and Power BI for seamless reporting and visualization. The final output aims to support policy-making, resource planning, and sustainable development initiatives.

# MARKDOWN ********************

# ## EDA

# CELL ********************

#Import libraries for data manipulation and visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from pyspark.sql.functions import col, lag
from pyspark.sql.window import Window
#Time Series Forecasting
from statsmodels.tsa.arima.model import ARIMA  # For building ARIMA models for time series forecasting

# Suppress unnecessary warnings to keep the notebook clean
from warnings import filterwarnings
filterwarnings('ignore')


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Load csv
modelling = spark.read.format("csv").option("header","true").load("Files/modelling.csv")
# df now is a Spark DataFrame containing CSV data from "Files/modelling.csv".
display(modelling)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#Listing numeric columns and converting them to float type
numeric_columns = [
    'Electricity_from_hydro_TWh',
    'Electricity_from_solar_TWh',
    'Other_renewables_including_bioenergy_TWh',
    'Electricity_generation_TWh',
    'Total_Renewable_Electricity_TWh',
    'Electricity_from_Non_Renewables_TWh',
    'Urbanization',
    'Electricity_from_wind_TWh'
]

# Apply the cast to each column
for col_name in numeric_columns:
    modelling = modelling.withColumn(col_name, col(col_name).cast("float"))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#Show schema of the DataFrame
modelling.printSchema()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Verify table creation
spark.sql("SHOW TABLES").show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#Create a Spark DataFrame with selected features and time series columns. Ive used PySpark
energy_df = spark.sql("""
SELECT 
    Entity,
    Year,
    `Electricity_from_wind_TWh`,
    `Electricity_from_hydro_TWh`,
    `Electricity_from_solar_TWh`,
    `Other_renewables_including_bioenergy_TWh`,
    `Electricity_generation_TWh`,
    `Total_Renewable_Electricity_TWh`,
    `Electricity_from_Non_Renewables_TWh`,
    Urbanization,
    -- Create time series features
    LAG(`Total_Renewable_Electricity_TWh`, 1) OVER (PARTITION BY Entity ORDER BY Year) AS prev_year_renewables,
    ROW_NUMBER() OVER (PARTITION BY Entity ORDER BY Year) AS time_index
FROM modelling
""")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#Handle nulls from lag function
energy_df = energy_df.fillna(0, subset=["prev_year_renewables"])
display(energy_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#Creating ML-ready features
final_features = energy_df.withColumn(
    "urbanization_growth", 
    col("Urbanization") - lag("Urbanization", 1).over(Window.partitionBy("Entity").orderBy("Year"))
).withColumn(
    "renewables_ratio",
    col("Total_Renewable_Electricity_TWh") / col("Electricity_generation_TWh")
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#Saving as Delta table for MLflow access
final_features.write.mode("overwrite").saveAsTable("energy_ml_features")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#importing pyspark libraries for compatibility with fabric
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
import mlflow

#Preparing features with time lags
window_spec = Window.partitionBy("Entity").orderBy("Year")

feature_df = energy_df.withColumn("lag1", lag("Total_Renewable_Electricity_TWh", 1).over(window_spec)) \
                    .withColumn("lag2", lag("Total_Renewable_Electricity_TWh", 2).over(window_spec)) \
                    .na.drop()

# test/train split
(train_data, test_data) = feature_df.randomSplit([0.8, 0.2], seed=42)

# Creating and training pipeline
assembler = VectorAssembler(
    inputCols=["lag1", "lag2", "Urbanization"],
    outputCol="features"
)

lr = LinearRegression(featuresCol="features", labelCol="Total_Renewable_Electricity_TWh")
pipeline = Pipeline(stages=[assembler, lr])
model = pipeline.fit(train_data)

# Generating predictions
predictions = model.transform(test_data)

# Evaluation and log metrics
evaluator = RegressionEvaluator(labelCol="Total_Renewable_Electricity_TWh")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("features", assembler.getInputCols())
    
    # Calculate and log metrics
    metrics = {
        "rmse": evaluator.evaluate(predictions, {evaluator.metricName: "rmse"}),
        "r2": evaluator.evaluate(predictions, {evaluator.metricName: "r2"}),
        "mae": evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
    }
    
    for metric, value in metrics.items():
        mlflow.log_metric(metric, value)
    
    # Save model
    mlflow.spark.log_model(model, "energy_forecaster")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#Import functions
from pyspark.sql.functions import col, explode, array, lit, create_map
from itertools import chain

# Define energy types
energy_types = [
    'Electricity_from_wind_TWh',
    'Electricity_from_hydro_TWh',
    'Electricity_from_solar_TWh',
    'Other_renewables_including_bioenergy_TWh',
    'Electricity_generation_TWh',
    'Total_Renewable_Electricity_TWh',
    'Electricity_from_Non_Renewables_TWh'
]

# Convert the dataframe to long format from wide
long_df = energy_df.select(
    "Entity", 
    "Year", 
    "Urbanization",
    explode(
        create_map(
            *list(chain(*[  # Flatten the list of key-value pairs
                (lit(col_name), col(col_name)) for col_name in energy_types
            ])))
    ).alias("EnergyType", "Value")
)

display(long_df)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#import libraries
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, count

# Ensure each series has enough historical data points
MIN_HISTORY_YEARS = 5

#Define window for time ordering
window_spec = Window.partitionBy('Entity', 'EnergyType').orderBy('Year')

#Add row numbers, count records per group, filter those with enough data, and join back
forecast_ready = long_df.withColumn('row_num', row_number().over(window_spec)) \
    .groupBy('Entity', 'EnergyType') \
    .agg(count('*').alias('count')) \
    .filter(col('count') >= MIN_HISTORY_YEARS) \
    .join(long_df, ['Entity', 'EnergyType'])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#import libraries
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Define output schema for forecast results
result_schema = StructType([
    StructField('Entity', StringType()),
    StructField('EnergyType', StringType()),
    StructField('Year', IntegerType()),
    StructField('Forecast', DoubleType()),
    StructField('Lower_CI', DoubleType()),
    StructField('Upper_CI', DoubleType())
])

@pandas_udf(result_schema, functionType=PandasUDFType.GROUPED_MAP)
def forecast_energy(group):
    """
    ARIMA forecasting UDF that takes a grouped dataframe and returns forecasts
    """
    try:
        # Sort by year and prepare data
        group = group.sort_values('Year')
        ts_data = group.set_index('Year')['Value']
        
        # Fit ARIMA model
        model = ARIMA(ts_data, order=(1,1,1))
        model_fit = model.fit()
        
        # Generate 5-year forecast
        forecast_years = 5
        forecast = model_fit.get_forecast(steps=forecast_years)
        ci = forecast.conf_int()
        
        # Create result DataFrame
        future_years = range(ts_data.index.max()+1, ts_data.index.max()+1+forecast_years)
        result = pd.DataFrame({
            'Entity': [group['Entity'].iloc[0]] * forecast_years,
            'EnergyType': [group['EnergyType'].iloc[0]] * forecast_years,
            'Year': future_years,
            'Forecast': forecast.predicted_mean.values,
            'Lower_CI': ci.iloc[:,0].values,
            'Upper_CI': ci.iloc[:,1].values
        })
        
        return result
        
    except Exception as e:
        # Return empty DataFrame for series that fail
        print(f"Forecast failed for {group['Entity'].iloc[0]} - {group['EnergyType'].iloc[0]}: {str(e)}")
        return pd.DataFrame(columns=['Entity', 'EnergyType', 'Year', 'Forecast', 'Lower_CI', 'Upper_CI'])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Applying forecasting UDF to each group
forecast_results = forecast_ready.groupby('Entity', 'EnergyType').apply(forecast_energy)

# Cache the results as this computation can be expensive
forecast_results.cache()

# Show 20 results
display(forecast_results.limit(20))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Save forecasts to a Delta table
forecast_results.write.mode("overwrite").saveAsTable("energy_forecasts")

# For visualization, convert to pandas
sample_forecasts = forecast_results.filter(
    (col('Entity') == 'Kenya') & 
    (col('EnergyType') == 'Total_Renewable_Electricity_TWh')
).toPandas()

# Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(sample_forecasts['Year'], sample_forecasts['Forecast'], label='Forecast')
plt.fill_between(
    sample_forecasts['Year'],
    sample_forecasts['Lower_CI'],
    sample_forecasts['Upper_CI'],
    alpha=0.2,
    label='95% CI'
)
plt.title('Kenya Renewable Electricity Forecast')
plt.xlabel('Year')
plt.ylabel('TWh')
plt.legend()
plt.grid(True)
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# For the above Ive used Kenya just as an example to test whether it works.

# CELL ********************

# Convert Spark DataFrame to a pandas DataFrame that Power BI can use
forecast_pd = forecast_results.select(
    "Entity", 
    "EnergyType", 
    "Year", 
    "Forecast", 
    "Lower_CI", 
    "Upper_CI"
).toPandas()

# Add a column for the time period type (historical vs forecast)
import numpy as np
current_year = dt.datetime.now().year
forecast_pd['Period'] = np.where(
    forecast_pd['Year'] <= current_year, 
    'Historical', 
    'Forecast'
)

# Save to CSV for Power BI
forecast_pd.to_csv('energy_forecasts.csv', index=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### Conclusion

# MARKDOWN ********************

# In this notebook, we successfully processed and prepared energy data for forecasting using PySpark and ARIMA. After transforming the dataset into a long format, we ensured each energy type had sufficient historical data points for accurate modeling. By applying an ARIMA model through a custom Pandas UDF, we were able to generate forecasts for multiple energy types, providing valuable insights into future trends. The results, including the predicted values and confidence intervals, offer a reliable foundation for decision-making and strategic planning in the energy sector.
# 
# However, it’s important to note that our exploration was limited by the F2 SKU capacity, which constrained the depth of our analysis and the scalability of the model. Despite these limitations, the methodology used in this notebook demonstrates the potential of combining big data processing with statistical modeling to tackle complex forecasting challenges.

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
