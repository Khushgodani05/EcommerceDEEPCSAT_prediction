from sklearn.base import BaseEstimator, TransformerMixin
import textblob
from textblob import TextBlob
import pandas as pd
import torch 
from torch.utils.data import Dataset


class FeatureBuilder(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self,X):

        X = X.copy()

        X["customerremarks"] = X["customerremarks"].fillna("")

        X["sentiment_score"] = X["customerremarks"].apply(
            lambda text: TextBlob(str(text)).sentiment.polarity
        )

        X["issuereportedat"] = pd.to_datetime(X["issuereportedat"], errors="coerce")
        X["issueresponded"] = pd.to_datetime(X["issueresponded"], errors="coerce")
        X["surveyresponsedate"] = pd.to_datetime(X["surveyresponsedate"], errors="coerce")

        X["issuemonth"] = X["issuereportedat"].dt.month.astype(float)
        X["issuerespondedmonth"] = X["issueresponded"].dt.month.astype(float)

        X["responsehours"] = (
            (X["issueresponded"] - X["issuereportedat"]).dt.total_seconds() / 3600
        )

        X["responsehourscategory"] = pd.cut(
            X["responsehours"],
            bins=[0,6,12,18,24,3000],
            labels=["within 6","6-12","12-18","18-24","24+"]
        )

        X = X.drop(
            [
                "issuereportedat",
                "issueresponded",
                "surveyresponsedate",
                "supervisor",
                "manager",
                "responsehours",
                "customerremarks"
            ],
            axis=1
        )

        return X
