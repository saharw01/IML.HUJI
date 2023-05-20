from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, y_encoded, classes_counts = np.unique(y, return_inverse=True, return_counts=True)
        self.pi_ = classes_counts / len(y)

        X_y = np.c_[X, y]
        X_y = X_y[X_y[:, -1].argsort()]
        samples_grouped_by_class = np.split(X_y[:, :-1], np.unique(X_y[:, -1], return_index=True)[1][1:])

        self.mu_ = np.array([np.sum(class_k_samples, axis=0) / classes_counts[k]
                             for k, class_k_samples in enumerate(samples_grouped_by_class)])

        centered_samples = X - self.mu_[y_encoded]
        self.cov_ = centered_samples.T @ centered_samples / (len(X) - len(self.classes_))
        self._cov_inv = inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        # According to a claim proven in the Tirgul, the chosen implementation is equivalent to:
        # return self.classes_[np.argmax(self.likelihood(X), axis=1)]
        b = np.log(self.pi_) - .5 * np.einsum("ki,ij,kj->k", self.mu_, self._cov_inv, self.mu_)
        a = self.mu_ @ self._cov_inv
        return self.classes_[np.argmax((X @ a.T) + b, axis=1)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        X_class_centered = X[:, np.newaxis, :] - self.mu_
        likelihood = (np.exp(-.5 * np.einsum("nkj,ji,nki->nk", X_class_centered, self._cov_inv, X_class_centered))
                      / np.sqrt(det(self.cov_) * (2 * np.pi) ** X.shape[1]))
        return likelihood * self.pi_  # likelihood * prior

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))
