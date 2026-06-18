# Disney World Wait Time Predictor

Predicting Walt Disney World ride wait times using historical queue data

## Overview
This project analyzes and mdoels wait times across 26 Disney World rides using data from 2015-2021. It explores how different factors such as time of day, season, temperature, and rainfall affect how long guests wait in line.

## Data
- Ride wait times: posted and actual wait times per ride per hour
- Weather: daily temperature and precipitation from Kissimmee, FL
- Metadata: Disney park seasons/holidays calendar
- Raw data not included in this repo due to size, but can be found here: https://drive.google.com/drive/folders/10vja2BRC6dOKRijGzBRGuL99LhjXdDsX?usp=share_link

## Features Used
- Hour of day, day of week, month, season
- Temperature (mean, max, min), precipitation
- Cyclical encoding (sin/cos) for time
- Ride identity (one-hot encoded)

## Models
- Linear Regression
- Random Forest
- XGBoost

## Results
- Wait times ranged from 0 to 390 minutes, with a mean of 40 minutes and median of 30 minutes
- Flight of Passage had the highest average wait time and variability (std: 53.6 min), followed by Space Mountain and 7 Dwarfs Train
- Wait times peak mid-day and drop off in the early morning and late evening
- Weekends see higher wait times than weekdays, with Saturday being the busiest day
- Christmas peak was the busiest season (avg 63 min), while September Low and Halloween were the least crowded (avg ~29-31 min)
- Temperature and precipitation showed weak correlations with wait times individually, suggesting crowd patterns are driven more by calendar/season than weather

### Model Performance
- Linear Regression: MAE: 40.53 min RMSE: 44.50 min
- Random Forest: MAE: 22.60 min RMSE: 27.69 min
- XGBoost: MAE: 24.19 min RMSE: 28.08 min

### Feature Importance
Most predictive features: ride identity, hour of day, year, and day of year
