# Last updated December 14, 2023
# Version 0.1.0

import pandas as pd
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from category_encoders.one_hot import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector


class CleaningTransformer(BaseEstimator, TransformerMixin):
    """Cleaning transformer class of stroke dataset."""

    def fit(self, X: pd.DataFrame, y=None):
        """Do not change anything during fitting."""
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Apply necessary cleaning and feature engineering to the data."""
        X_new = X.copy()
        X_new["gender"] = X_new["gender"].str.lower()
        X_new.loc[X_new["gender"] == "other", ["gender"]] = np.nan
        X_new["smoking_status"] = (
            X_new["smoking_status"].str.lower().str.replace(" ", "_")
        )
        X_new.loc[
            (X_new["smoking_status"] == "unknown") & (X_new["age"] < 10),
            ["smoking_status"],
        ] = "never_smoked"
        X_new.loc[X_new["smoking_status"] == "unknown", ["smoking_status"]] = (
            np.nan
        )
        X_new["ever_smoked"] = np.nan
        X_new.loc[
            X_new["smoking_status"] == "never_smoked", ["ever_smoked"]
        ] = "No"
        X_new.loc[
            X_new["smoking_status"] == "formerly_smoked", ["ever_smoked"]
        ] = "Yes"
        X_new.loc[X_new["smoking_status"] == "smokes", ["ever_smoked"]] = "Yes"
        X_new = X_new.drop(columns="smoking_status")
        X_new = X_new.drop(columns="work_type")
        X_new = X_new.drop(columns="ever_married")
        return X_new


def build_full_pipe(model: sklearn.base.BaseEstimator) -> Pipeline:
    """Build full pipe (with cleaning and transformations) for stroke dataset
    with defined estimator."""
    binary_transformer = Pipeline(
        steps=[("impute", SimpleImputer(strategy="constant", fill_value=0))]
    )
    numerical_transformer = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value=0)),
            ("scale", None),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[(
            "onehot",
            OneHotEncoder(
                handle_unknown="value",
                handle_missing="ignore",
                use_cat_names=True,
            ),
        )]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_transformer,
                make_column_selector(dtype_include=float),
            ),
            (
                "bin",
                binary_transformer,
                make_column_selector(dtype_include=int),
            ),
            (
                "cat",
                categorical_transformer,
                make_column_selector(dtype_include=object),
            ),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    pipe = Pipeline(
        steps=[
            ("cleaner", CleaningTransformer()),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipe
