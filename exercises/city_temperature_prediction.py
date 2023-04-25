from typing import Tuple

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def lower_upper(x: pd.Series) -> Tuple[float, float]:
    """
    Calculate lower and upper bounds for outlier detection,
    i.e. a sample not in the interval [lower, upper] will be considered an outlier.
    (current method used is the standard deviation method)
    Parameters
    ----------
    x: pandas.Series
        samples of a single feature
    Returns
    -------
    lower and upper bounds for outlier detection
    """
    mu = x.mean()
    sigma = x.std()
    return mu-3*sigma, mu+3*sigma


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    res = pd.read_csv(filename, parse_dates=["Date"])
    res = res.dropna().drop_duplicates()
    res["DayOfYear"] = res["Date"].dt.dayofyear
    lower, upper = lower_upper(res["Temp"])
    res = res[(lower <= res["Temp"]) & (res["Temp"] <= upper)]
    res["Year"] = res["Year"].astype(str)  # convert to string so that color scale will be discrete in plots
    return res


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_df = df[df["Country"] == "Israel"]

    px.scatter(israel_df, x="DayOfYear", y="Temp", color="Year",
               title="Israel Temperature as Function of Days of The Year"
               ).write_html("Israel_temp_over_day_of_year.html")

    px.bar(israel_df.groupby("Month", as_index=False).agg(temp_std=("Temp", "std")),
           x="Month", y="temp_std", title="Israel Temperature Standard Deviation per Month"
           ).write_html("Israel_temp_std_per_month.html")

    # Question 3 - Exploring differences between countries
    px.line(df.groupby(["Month", "Country"], as_index=False).agg(temp_std=("Temp", "std"), temp_avg=("Temp", "mean")),
            x="Month", y="temp_avg", error_y="temp_std", color="Country",
            title="Average Temperature as Function of Month"
            ).write_html("avg_temp_over_month_per_country.html")

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(israel_df["DayOfYear"], israel_df["Temp"])

    losses = {}
    for k in range(1, 11):
        losses[k] = PolynomialFitting(k).fit(train_X, train_y).loss(test_X, test_y)
        losses[k] = np.round(losses[k], 2)
    losses = pd.DataFrame(losses.items(), columns=["poly_deg", "loss"])

    print(losses)
    px.bar(losses, x="poly_deg", y="loss", text="loss",
           title="Israel Models MSE Over Test Set per Polynomial Model Degree"
           ).write_html("Israel_temp_prediction_mse_per_poly_deg.html")

    # Question 5 - Evaluating fitted model on different countries
    k = int(losses.iloc[losses["loss"].idxmin()]["poly_deg"])
    israel_model = PolynomialFitting(k).fit(israel_df["DayOfYear"], israel_df["Temp"])

    losses = {}
    for country in df["Country"].unique():
        country_df = df[df["Country"] == country]
        losses[country] = israel_model.loss(country_df["DayOfYear"], country_df["Temp"])
        losses[country] = np.round(losses[country], 2)
    losses = pd.DataFrame(losses.items(), columns=["country", "loss"])

    px.bar(losses, x="country", y="loss", color="country", text="loss",
           title="Israel Model MSE Over Entire Data per Country"
           ).write_html("temp_prediction_mse_per_country.html")
