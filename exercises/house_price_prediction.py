from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


FEATURES_NOT_USED_IN_MODEL = ["id", "lat", "long", "sqft_living15", "sqft_lot15", ]
mean_values = {}


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    res = pd.concat([X, y], axis=1)
    res = res.drop(columns=FEATURES_NOT_USED_IN_MODEL)
    res["date"] = res["date"].str[0:4]  # keep only the year the house was sold

    if y is not None:  # Processing Training Data:
        res = res.dropna().drop_duplicates()
        res["date"] = res["date"].astype(int)

        res = res[res["waterfront"].isin([0, 1]) &
                  res["view"].isin(range(5)) &
                  res["condition"].isin(range(1, 6)) &
                  res["grade"].isin(range(1, 14))]

        for feature in ["price", "sqft_living", "sqft_lot", ]:
            res = res[res[feature] > 0]
        for feature in ["sqft_above", "sqft_basement", "bedrooms", "bathrooms", "floors", "yr_renovated", "yr_built", ]:
            res = res[res[feature] >= 0]

        res = res[(res["yr_built"] <= res["date"]) & (res["yr_renovated"] <= res["date"])]

        for feature in res:
            mean_values[feature] = res[feature].mean()
    else:  # Processing Test Data
        res = res.fillna(value=mean_values)
        res["date"] = res["date"].astype(int)

    res["age"] = res.apply(lambda row: row["date"] - row["yr_built"], axis=1)
    res["age_renovated"] = res.apply(
        lambda row: row["age"] if row["yr_renovated"] == 0 else row["date"] - row["yr_renovated"], axis=1
    )
    res = res.drop(columns=["date", "yr_built", "yr_renovated"])

    res["zipcode"] = res["zipcode"].astype(int)
    res = pd.get_dummies(res, columns=["zipcode"], prefix=["zipcode"])

    return res


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    X_y = pd.concat([X, y], axis=1).dropna()
    X, y = X_y.iloc[:, :-1], X_y.iloc[:, -1]
    for feature in X:
        f_col = X[feature].astype(np.float64)
        corr = np.cov(f_col, y)[0, 1] / (np.std(f_col) * np.std(y))
        px.scatter(pd.DataFrame({"x": f_col, "y": y}), x="x", y="y",
                   title=f"Response as Function of {feature}<br>"
                         f"Pearson Correlation between {feature} and response: {corr}",
                   labels={"x": f"{feature}", "y": "Response"}, trendline="ols"
                   ).write_html(output_path + f"/{feature}_pearson_correlation_with_response.html")


if __name__ == "__main__":
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    df = df[df["price"] > 0]  # drop rows with missing price

    # Question 1 - split data into train and test sets
    train_X, train_y, test_X, test_y = split_train_test(df.drop(columns=["price"]), df["price"])

    # Question 2 - Preprocessing of housing prices dataset
    train_X = preprocess_data(train_X, train_y)
    train_X, train_y = train_X.drop(columns=["price"]), train_X["price"]
    test_X = preprocess_data(test_X)
    test_X = test_X.reindex(columns=train_X.columns, fill_value=0)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(df[["sqft_living", "condition"]], df["price"])

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    percentages = list(range(10, 101))
    losses = np.zeros((len(percentages), 10))
    for i, p in enumerate(percentages):
        for j in range(losses.shape[1]):
            train_X_p_sample = train_X.sample(frac=p / 100.0)
            train_y_p_sample = train_y.loc[train_X_p_sample.index]
            losses[i, j] = LinearRegression(include_intercept=True).fit(train_X_p_sample, train_y_p_sample)\
                .loss(test_X, test_y)

    mean = losses.mean(axis=1)
    std = losses.std(axis=1)

    go.Figure([go.Scatter(x=percentages, y=mean, mode="markers+lines"),
               go.Scatter(x=percentages, y=mean - 2 * std, mode="lines", line=dict(color="lightgrey")),
               go.Scatter(x=percentages, y=mean + 2 * std, fill="tonexty", mode="lines", line=dict(color="lightgrey"))],
              layout=go.Layout(title="MSE Over Test Set as Function of Training Size",
                               xaxis_title="Percentage of Training Set",
                               yaxis_title="MSE Over Test Set",
                               showlegend=False)
              ).write_html("house_price_prediction_mse_over_training_percentage.html")
