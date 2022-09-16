import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


@st.cache(allow_output_mutation=True)
def file_upload(sheet_file):
    try:
        if sheet_file.name.endswith("csv") or sheet_file.name.endswith("txt"):

            df = pd.read_csv(sheet_file)
            return df
        elif sheet_file.name.endswith("xlsx"):
            df = pd.read_excel(sheet_file)
            return df

    except Exception as e:
        st.exception(e)


def estimate_trend(easting, northing, observations, degree):
    """
    Estimate a 2D polynomial regional trend that fits the observations.
    Returns the estimated trend values.
    """
    npoints = observations.size
    coordinates = np.empty((npoints, 2))
    coordinates[:, 0] = easting
    coordinates[:, 1] = northing

    features = PolynomialFeatures(degree)
    X = features.fit_transform(coordinates)

    polynomial = LinearRegression()
    polynomial.fit(X, observations)

    trend = polynomial.predict(X)
    return trend
