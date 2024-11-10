# Production Forecast Time Series analysis
This repo contains code for production forecast using Time series analysis.

## Project Overview

Time series analysis is crucial in many fields, such as finance, economics, environmental science, and more, where understanding the patterns over time can help with forecasting and decision-making. This notebook introduces the fundamental steps in analyzing and visualizing time series data.

## Prerequisites

Before running the notebook, ensure that the following Python libraries are installed:

- `pandas`: For data manipulation and analysis.
- `matplotlib`: For data visualization.
- `statsmodels`: For advanced statistical models and time series decomposition.

To install these libraries, you can use:

```bash
pip install pandas matplotlib statsmodels
```
## Usage

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/PerceptiaAI/Production-Forecast-Time-series-Predictive-model-.git
cd Production-Forecast-Time-series-Predictive-model
```

## Example Code Snippet

The following code snippet demonstrates how to decompose a time series into its components:

```python
from statsmodels.tsa.seasonal import seasonal_decompose 

result = seasonal_decompose(data['value'], model='additive', period=12)
result.plot()
plt.show()
```
## Graphs
### 1. Line Plot of Time Series Data

- **Description**: This plot displays the raw time series data over time.
- **Purpose**: Line plots help in visualizing trends and patterns over time. For time series analysis, this is usually the first step, as it allows you to identify whether there are any obvious trends (increasing or decreasing patterns) and seasonality (repeating cycles).
- **Insights**: From the line plot, you can tell whether the data has an upward or downward trend, and you may also detect periodic patterns if there’s any seasonality.

![Dataset](https://github.com/PerceptiaAI/Production-Forecast-Time-series-Predictive-model-/blob/main/Images/dataset.png)

### 2. Decomposition Plot

**Description:**  
This is a multiple-panel plot created using seasonal decomposition, where the time series is broken down into three components:
- **Trend:** Displays the long-term increase or decrease in the data.
- **Seasonal:** Shows repeating patterns within a fixed period (e.g., monthly or yearly seasonality).
- **Residual:** Shows what’s left after removing the trend and seasonal components; often referred to as "noise."

**Purpose:**  
Decomposition allows you to understand the different factors influencing the time series. Each component isolates a specific aspect:
- The **trend** shows the direction of the series (e.g., increasing sales over time).
- The **seasonal** component highlights any recurring patterns (e.g., higher sales in December).
- The **residual** helps identify irregularities or outliers.

**Insights:**  
This breakdown is useful in identifying if the data has underlying trends or seasonality, which can aid in choosing the right forecasting models.

![Decmposition](https://github.com/PerceptiaAI/Production-Forecast-Time-series-Predictive-model-/blob/main/Images/decompose%20data.png)

## Forecast

![Forecast](https://github.com/PerceptiaAI/Production-Forecast-Time-series-Predictive-model-/blob/main/Images/Forecast.png)

![Predictions](https://github.com/PerceptiaAI/Production-Forecast-Time-series-Predictive-model-/blob/main/Images/Screenshot%202024-11-11%20011744.png)
