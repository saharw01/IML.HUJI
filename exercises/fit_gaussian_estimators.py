from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    mu, sigma, num_samples = 10, 1, 1000

    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(mu, sigma, num_samples)
    gaussian = UnivariateGaussian().fit(X)
    print((gaussian.mu_, gaussian.var_))

    # Question 2 - Empirically showing sample mean is consistent
    mean_deviation = []
    sample_sizes = np.arange(start=10, stop=num_samples, step=10, dtype=int)
    for n in sample_sizes:
        mean_deviation.append(np.abs(mu - UnivariateGaussian().fit(X[:n]).mu_))
    go.Figure(go.Scatter(x=sample_sizes, y=mean_deviation, mode="markers+lines"),
              go.Layout(title=r"$\text{Deviation of Sample Mean Estimator As Function of Sample Size}$",
                        xaxis_title=r"$\text{Sample Size}$",
                        yaxis_title=r"$\text{Estimator Deviation}$")).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = gaussian.pdf(X)
    go.Figure(go.Scatter(x=X, y=pdfs, mode="markers"),
              go.Layout(title=r"$\text{Empirical pdf under fitted model}$",
                        xaxis_title=r"$\text{Samples}$",
                        yaxis_title=r"$N(\hat{\mu},\hat{\sigma}^2)$")).show()


def test_multivariate_gaussian():
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    num_samples = 1000

    # Question 4 - Draw samples and print fitted model
    X = np.random.multivariate_normal(mu, cov, num_samples)
    gaussian = MultivariateGaussian().fit(X)
    print(gaussian.mu_)
    print(gaussian.cov_)

    # Question 5 - Likelihood evaluation
    num_args = 200
    args = np.linspace(-10, 10, num_args)
    evaluations = np.zeros((num_args, num_args))
    for i, f1 in enumerate(args):
        for j, f3 in enumerate(args):
            evaluations[i, j] = MultivariateGaussian.log_likelihood(
                np.array([f1, 0, f3, 0]), cov, X
            )
    go.Figure(go.Heatmap(x=args, y=args, z=evaluations),
              go.Layout(title=r"$\text{Log-Likelihood of Multivariate Gaussian As Function of Expectation }"
                              r"{\mu=[f_1, 0, f_3, 0]}$",
                        xaxis_title=r"$f_3$",
                        yaxis_title=r"$f_1$")).show()

    # Question 6 - Maximum likelihood
    ind_of_max = np.unravel_index(np.argmax(evaluations), evaluations.shape)
    print((np.round(args[ind_of_max[0]], 3), np.round(args[ind_of_max[1]], 3)))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
