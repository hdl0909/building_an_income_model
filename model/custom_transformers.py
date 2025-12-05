from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class DropHighNaN(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.cols_ = None

    def fit(self, X, y=None):
        self.cols_ = X.columns[X.isnull().mean() < self.threshold]
        return self

    def transform(self, X):
        return X[self.cols_].copy()


class SplitObjectColumns(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.categorical_cols_ = None

    def fit(self, X, y=None):
        categorical = []
        for col in X.select_dtypes(include=['object']).columns:
            try:
                pd.to_numeric(X[col].dropna())
            except ValueError:
                categorical.append(col)
        self.categorical_cols_ = categorical
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.categorical_cols_:
            X[col] = X[col].astype(str).fillna("nan")
        return X


class FillNumericMedian(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.medians_ = {}

    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            self.medians_[col] = X[col].median()
        return self

    def transform(self, X):
        X = X.copy()
        for col, median in self.medians_.items():
            X[col] = X[col].fillna(median)
        return X

class KeepSelectedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features
        self.features_ = None
    def fit(self, X, y=None):
        self.features_ = self.features
        return self
    def transform(self, X):
        return X[self.features_].copy()