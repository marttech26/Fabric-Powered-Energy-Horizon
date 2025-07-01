# **Fabric Powered Energy-Horizon**

![Power plants](power_plants_background.jpg)

## **Overview**

The **Fabric Powered Energy Horizon** project leverages the power of Microsoft Fabric and advanced statistical modeling to predict the future trends in energy production. This project specifically focuses on forecasting renewable and non-renewable electricity generation based on historical data. By utilizing a robust ARIMA time series forecasting model, we aim to provide actionable insights into the evolution of energy production over the next coming years.

This solution utilizes **PySpark** for large-scale data processing and **ARIMA** for time series forecasting, offering a scalable approach to energy prediction that can be adapted to other regions or datasets.

## **Key Features**

- **Energy Forecasting:** Predicts the future energy generation (renewable and non-renewable) using ARIMA.
- **Time Series Analysis:** Uses historical data to model trends and generate forecasts.
- **Microsoft Fabric Integration:** Leverages Microsoft Fabric for efficient data processing and seamless integration with cloud services.
- **Confidence Intervals:** Provides forecast values along with lower and upper confidence intervals for better decision-making.
- **Scalable:** Built to handle large datasets using PySpark, making it suitable for various energy production datasets across different regions.

## **Technologies Used**

- **Microsoft Fabric:** Provides the platform for large-scale data processing and analysis.
- **PySpark:** Handles distributed data processing and transformation at scale.
- **ARIMA (AutoRegressive Integrated Moving Average):** Time series forecasting model used for predicting future energy trends.
- **Pandas UDFs (User Defined Functions):** For parallel processing of forecasts across different entities and energy types.
- **Python Libraries:** 
  - `pandas` for data manipulation.
  - `statsmodels` for ARIMA time series forecasting.

## **Setup and Installation**

### **Prerequisites**

- Python 3.x
- Spark 3.x (with PySpark)
- Microsoft Fabric (for cloud integration)
- ARIMA module from `statsmodels`
- Libraries: pandas, statsmodels, pyspark

### **Install Required Packages**

You can install the necessary Python libraries using `pip`:

```bash
pip install pandas matplotlib seaborn statsmodels pyspark
