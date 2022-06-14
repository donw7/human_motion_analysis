"""
Custom pipeline data transformations
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import signal

def get_indices(df):
  df_match = df['path'].eq(df['path'].shift())
  row_indices = np.where(df_match.values == False)[0]
  row_indices = np.append(row_indices, len(df_match))
  row_indices_enumerated = []
  for i in range(1, len((row_indices))):
    idx = np.arange(row_indices[i - 1], row_indices[i])
    row_indices_enumerated.append(idx)

  return row_indices_enumerated # list of np.arrays

class NormalizeDistanceBodyLength(BaseEstimator, TransformerMixin):
  """
  normalizes coordinates to body length by finding min and max y distance of specified head and feet columns and then dividing each value of other columns
  
  Expects df and specify:
  select_colnames: list of columns to normalize
  head_colnames: list of columns to find max y distance (mean to account for e.g. unilateral occlusions)
  feet_colnames: list of columns to find min y distance (mean to account for e.g. unilateral occlusions)
  
  Returns a np.array of normalized values
  """
  def __init__(self, select_colnames, head_colnames, feet_colnames, row_indices_enumerated):
      self.select_colnames = select_colnames
      self.head_colnames = head_colnames
      self.feet_colnames = feet_colnames
      self.row_indices_enumerated = row_indices_enumerated

  def fit(self, X_df, X_out=None):
    return self

  def transform(self, X_df):
    max_head_values = []
    max_feet_values = []

    for indices in self.row_indices_enumerated:
      max_head_value = np.mean(np.max(X_df.iloc[indices][self.head_colnames].values, axis=0))
      min_feet_value = np.mean(np.min(X_df.iloc[indices][self.feet_colnames].values, axis=0))
      y_body_length = np.abs(max_head_value - min_feet_value)
      X_out = X_df[self.select_colnames].applymap(lambda x: x / y_body_length).to_numpy()


    return X_out # np.array


class MedFilter(BaseEstimator, TransformerMixin):
  """
  column-wise, video-wise (as opposed to population-wise) median filter
  expects input row_indices_enumerated externally
  """
  def __init__(self, row_indices_enumerated, kernel_size=3):
    self.row_indices_enumerated = row_indices_enumerated
    self.kernel_size = kernel_size
    pass

  def fit(self, X, X_out=None):
    return self

  def transform(self, X):
    X_out = np.zeros(X.shape)
    for col in range(len(X.shape)):
      for indices in self.row_indices_enumerated:
        filtered_col = signal.medfilt(X[indices, col], self.kernel_size)
        X_out[indices, col] = filtered_col


    return X_out 


