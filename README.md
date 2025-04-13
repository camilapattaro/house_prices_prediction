# Predicting House Prices with Machine Learning

**Machine Learning** is a powerful tool for predicting house prices, helping buyers, sellers, and investors make smarter decisions in the real estate market. By analyzing a lot of data, ML models can give accurate, data-driven price estimates, making the market more transparent and helping with better financial planning.

In this beginner-friendly project, we’re going to predict house prices based on features like the number of bedrooms, bathrooms, floor level, square footage, and more. We’ll build a simple regression model using Python and libraries like Pandas, NumPy, Matplotlib, Seaborn, and Scikit-Learn. The plan is to do a quick exploratory data analysis and then move on to building, evaluating, and improving the model.

## The Dataset
The dataset we’ll be working with includes house sales from King County, which covers the Seattle area. It contains 21 features and over 21,000 rows, representing homes sold between May 2014 and May 2015. You can find the dataset [here](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction).

**id** — A notation for a house

**date** — Date house was sold

**price** — Price is prediction target

**bedrooms** — Number of bedrooms

**bathrooms** — Number of bathrooms

**sqft_living** —Square footage of the home

**sqft_lot** — Square footage of the lot

**floors** — Total floors (levels) in house

**waterfront** — House which has a view of a waterfront

**view** — Has been viewed

**condition** — How good the condition is overall

**grade** — overall grade given to the housing unit, based on King County grading system

**sqft_above** — Square footage of house apart from basement

**sqft_basement** — Square footage of the basement

**yr_built** — Built Year

**yr_renovated** — Year when house was renovated

**zipcode** — Zip code

**lat** — Latitude coordinate

**long** — Longitude coordinate

**sqft_living15** — Living room area in 2015 (implies some renovations)

**sqft_lot15** — LotSize area in 2015 (implies some renovations)

## Data Analysis and Model Building

**Codes are located in the *house_price_prediction.py* folder.**

## Conclusion
The R² improvement shows that the model got much better through the different steps (model development, pipeline, and evaluation).

By using Ridge Regression and applying a second-order polynomial transform during the evaluation, the model was able to capture the relationships between features and price more effectively, making it better at generalizing to new data.

An R² of approximately 0.82 is solid for predicting house prices, and it shows that the model is doing a good job of capturing the main factors affecting prices in the dataset.
