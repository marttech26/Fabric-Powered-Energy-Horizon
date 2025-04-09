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

# # 🌍 Fabric Powered Energy Horizon
# 
# ## 1. PROJECT OVERVIEW
# 
# ### (a) Introduction
# Renewable energy—spanning solar, wind, hydro and bioenergy, holds the key to decarbonizing our planet and reducing the negative impacts of fossil fuel dependency. As global urgency mounts around climate change and sustainable development, our hackathon project, **Fabric Powered Energy Horizon**, sets out to explore, analyze, and forecast renewable energy trends globally.
# 
# By harnessing the robust capabilities of **Microsoft Fabric**, we dive deep into global energy data to uncover the pace of renewable energy adoption and provide actionable insights. This solution is designed to guide policymakers, investors, and communities toward smarter energy decisions and support a just energy transition.
# 
# ---
# 
# ### (b) Business Understanding
# As countries shift towards cleaner energy, understanding the renewable energy landscape becomes vital. This project investigates global patterns of energy generation with a special focus on renewable sources, providing valuable insights into growth trends, regional disparities, and future potential.
# 
# By leveraging machine learning, Microsoft Fabric’s data capabilities, and Power BI dashboards, we aim to:
# - Empower governments and environmental agencies with data for planning and regulation.
# - Guide power companies in capacity planning and technology adoption.
# 
# ---
# 
# ### (c) Problem Statement
# Despite the availability of vast amounts of global energy data, there remains a lack of localized, actionable insights—especially in **developing regions like Africa**—where data forecasting and capacity planning are often underdeveloped.
# 
# Our project addresses this challenge by:
# - Providing a scalable, reproducible framework to forecast renewable energy production.
# - Translating historical data into insights that can influence energy investments and planning.
# - Tackling cloud capacity limitations (e.g., Microsoft Fabric’s F2 SKU) through lightweight yet effective modeling.
# 
# ---
# 
# ### (d) Research Questions
# -  How has electricity generation from renewable sources evolved over time?
# -  Which renewable energy source shows the most significant growth?
# -  How has the adoption of renewables impacted the global energy mix?
# -  Are there noticeable disparities in adoption between developed and developing regions?
# 
# ---
# 
# ### (e) Main Objective
# To develop a robust and efficient predictive model using **time series analysis** to forecast future renewable energy generation and assist in global energy planning.
# 
# ---
# 
# ### (f) Specific Objectives
# - To evaluate the historical contribution of each renewable source (solar, wind, hydro, etc.).
# - To identify the fastest-growing renewable sources across countries.
# - To uncover regional disparities and growth opportunities in energy generation.
# - To visualize renewable energy forecasts interactively using Power BI.
# - To enable 1–5 year forecasting (limited to 1 year due to F2 trial constraints).
# 
# ---
# 
# ### (g) Data Understanding
# Our dataset was sourced from [Our World in Data](https://ourworldindata.org/energy), spanning energy production metrics from 2000 to 2024.
# 
# **Data Columns:**
# - `Entity`: Country or region
# - `Year`: Year of measurement
# - `Electricity_from_wind_TWh`: Wind power generation (TWh)
# - `Electricity_from_hydro_TWh`: Hydroelectric power generation (TWh)
# - `Electricity_from_solar_TWh`: Solar power generation (TWh)
# - `Other_renewables_including_bioenergy_TWh`: Other renewable sources like biomass and geothermal (TWh)
# - `Electricity_from_Non_Renewables_TWh`: Power generated from fossil fuels and nuclear (TWh)
# - `Total_Renewable_Electricity_TWh`: Sum of all renewables (TWh)
# - `Electricity_generation_TWh`: Total electricity generation (TWh)
# - `Urbanization`: % of the population living in urban areas
# 
# The energy units are standardized in **Terawatt-hours (TWh)** for easy comparison across technologies and countries.
# 
# ---
# 
# ## Tools & Technologies Used
# - **Microsoft Fabric**: For data ingestion, transformation, and hosting
# - **PySpark**: For scalable data processing and feature engineering
# - **ARIMA (Statsmodels)**: For time series forecasting
# - **Power BI**: For data visualization and dashboarding
# - **Pandas, Python, SQL**: Supporting tools for analysis and transformation
# 
# ---
# 
# ## 🔍 Key Takeaways
# - Renewable energy forecasting can be achieved with limited cloud capacity using lightweight models.
# - Even with minimal computational power (F2 SKU), actionable insights are possible for stakeholders.
# - Our model and dashboard provide an accessible foundation for further development and scaling.
# 
# ---
# 
# ## 🎯 Alignment with Global Goals
# This project contributes directly to **UN Sustainable Development Goal 7**: _"Ensure access to affordable, reliable, sustainable and modern energy for all."_ By improving renewable energy forecasting, especially in underserved regions, we pave the way for smarter, greener energy infrastructure.
# 
# ---
# 
# ## 📦 Dataset Access
# The dataset used in this project is publicly available and can be accessed here: [Our World in Data - Energy](https://ourworldindata.org/energy)
# 
# ---
# 


# CELL ********************

#Loading 1st CSV

df = spark.read.format("csv").option("header","true").load("abfss://1cf87aea-aa0c-4a93-80a1-929e61be15e5@onelake.dfs.fabric.microsoft.com/486921fd-3eb3-47c0-b9ce-f7804a355198/Files/modern-renewable-prod.csv")
# df now is a Spark DataFrame containing CSV data from "abfss://1cf87aea-aa0c-4a93-80a1-929e61be15e5@onelake.dfs.fabric.microsoft.com/486921fd-3eb3-47c0-b9ce-f7804a355198/Files/modern-renewable-prod.csv".
display(df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Display the shape of the DataFrame
print(f"Rows: {df.count()}, Columns: {len(df.columns)}")

# Drop the unnecessary column 'Code'
df = df.drop('Code')

# Get a concise summary of the DataFrame
df.printSchema()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Loading 2nd CSV

df2 = spark.read.format("csv").option("header","true").load("abfss://1cf87aea-aa0c-4a93-80a1-929e61be15e5@onelake.dfs.fabric.microsoft.com/486921fd-3eb3-47c0-b9ce-f7804a355198/Files/electricity-generation - electricity-generation.csv")
# df now is a Spark DataFrame containing CSV data from "abfss://1cf87aea-aa0c-4a93-80a1-929e61be15e5@onelake.dfs.fabric.microsoft.com/486921fd-3eb3-47c0-b9ce-f7804a355198/Files/electricity-generation - electricity-generation.csv".
display(df2)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Drop the unnecessary column 'Code'
df2 = df2.drop('Code')

#checking nulls
from pyspark.sql.functions import col, sum as _sum, when

df2.select([
    _sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in df2.columns
]).show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Load 3rd CSV

df3 = spark.read.format("csv").option("header","true").load("abfss://1cf87aea-aa0c-4a93-80a1-929e61be15e5@onelake.dfs.fabric.microsoft.com/486921fd-3eb3-47c0-b9ce-f7804a355198/Files/urban-and-rural-population.csv")
# df now is a Spark DataFrame containing CSV data from "abfss://1cf87aea-aa0c-4a93-80a1-929e61be15e5@onelake.dfs.fabric.microsoft.com/486921fd-3eb3-47c0-b9ce-f7804a355198/Files/urban-and-rural-population.csv".
display(df3)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Drop the unnecessary column 'Code'
df3 = df3.drop('Code')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ##### Merging the three CSVs

# CELL ********************

# df and df have 2 similar columns, Entity and Year. We'll merge on the two columns
merge_df = pd.merge(df, df2, on=['Entity', 'Year'], how='inner')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
