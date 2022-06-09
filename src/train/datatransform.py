"""
Custom pipeline data transformations
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class NormalizeDistanceBodyLength(BaseEstimator, TransformerMixin):
  """
  normalizes coordinates to body length by finding min and max y distance of specified head and feet columns and then dividing each value of other columns
  
  Expects df and specify:
  select_colnames: list of columns to normalize
  head_colnames: list of columns to find max y distance (mean to account for e.g. unilateral occlusions)
  feet_colnames: list of columns to find min y distance (mean to account for e.g. unilateral occlusions)
  
  Returns a df of normalized values
  """
  def __init__(self, select_colnames, head_colnames, feet_colnames):
      self.select_colnames = select_colnames
      self.head_colnames = head_colnames
      self.feet_colnames = feet_colnames

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    max_head_values = []
    max_feet_values = []

    for colname in self.head_colnames:
      max_head_values.append(X[colname].max())
    for colname in self.feet_colnames:
      max_feet_values.append(X[colname].min())
    
    y_body_length = np.mean(max_head_values) - np.mean(max_feet_values)
      
    return X[self.select_colnames].applymap(lambda x: x / y_body_length)


