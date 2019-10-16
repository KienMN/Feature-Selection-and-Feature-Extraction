"""
Supervised Principal Components Analysis

Paper: Supervised Principal Component Analysis: Visualization, Classification and Regression on Subspaces and Submanifolds.

Reference: https://github.com/kumarnikhil936/Supervised-PCA-Python
"""

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigsh as ssl_eigsh
from time import clock

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn import utils
from sklearn.preprocessing import KernelCenterer, scale
from sklearn.metrics.pairwise import pairwise_kernels

"""
Input: training data matrix, X, testing data example, x, kernel matrix of target variable, L, and training data size, n.

Onput: Dimension reduced training and testing data, Z and z.

1: H ← I − (n^−1)ee^T

2: Q ← XHLHX^T

3: Compute basis: U ← eigenvectors of Q corresponding to the top d eigenvalues.

4: Encode training data: Z ← U^T X

5: Encode test example: z ← U^T x
"""

class SupervisedPCA(BaseEstimator, TransformerMixin):
  def __init__(self, n_components, kernel='linear', eigen_solver='auto', max_iterations=None, gamma=0,
              degree=3, coef0=1, alpha=1.0, tolerance=0, fit_inverse_transform=False):
    self._n_components = n_components
    self._gamma = gamma
    self._tolerance = tolerance
    self._fit_inverse_transform = fit_inverse_transform
    self._max_iterations = max_iterations
    self._degree = degree
    self._kernel = kernel
    self._eigen_solver = eigen_solver
    self._coef0 = coef0
    self._centerer = KernelCenterer()
    self._alpha = alpha
  
  def _get_kernel(self, X, Y=None):
    """
    Returns a kernel matrix K such that K_{i, j} is the kernel between the ith and jth vectors of the given matrix X, if y is None.

    If y is not None, then K_{i, j} is the kernel between the ith array from X and the jth array from Y.

    Valid kernels are 'linear', 'rbf', 'poly', 'sigmoid', 'precomputed'
    """
    kwargs = {'gamma': self._gamma, 'degree': self._degree, 'coef0': self._coef0}
    return pairwise_kernels(X, Y, metric=self._kernel, n_jobs=-1, filter_params=True, **kwargs)

  def _fit(self, X, y):
    # Calculate kernel matrix of the labels Y and centre it and call it K (=H.L.H)
    K = self._centerer.fit_transform(self._get_kernel(y.reshape(-1, 1)))
  
  # deciding on the number of components to use
    if self._n_components is not None:
      n_components = min(K.shape[0], self._n_components)
    else:
      n_components = K.shape[0]
    
    # Scale X
    # scaled_X = scale(X)
    
    # calculate the eigen values and eigen vectors for X^T.K.X
    Q = (X.T).dot(K).dot(X)
    
    # If n_components is much less than the number of training samples, 
    # arpack may be more efficient than the dense eigensolver.
    if (self._eigen_solver=='auto'):
      if (Q.shape[0]/n_components) > 20:
        eigen_solver = 'arpack'
      else:
        eigen_solver = 'dense'
    else:
      eigen_solver = self._eigen_solver
    
    if eigen_solver == 'dense':
      # Return the eigenvalues (in ascending order) and eigenvectors of a Hermitian or symmetric matrix.
      self._lambdas, self._alphas = linalg.eigh(Q, eigvals=(Q.shape[0] - n_components, Q.shape[0] - 1))
      # argument eigvals = Indexes of the smallest and largest (in ascending order) eigenvalues
    
    elif eigen_solver == 'arpack':
      # deprecated :: self._lambdas, self._alphas = utils.arpack.eigsh(A=Q, n_components, which="LA", tol=self._tolerance)
      self._lambdas, self._alphas = ssl_eigsh(A=Q, k=n_components, which="LA", tol=self._tolerance)
        
    indices = self._lambdas.argsort()[::-1]
    
    self._lambdas = self._lambdas[indices]
    self._lambdas = self._lambdas[self._lambdas > 0]  # selecting values only for non zero eigen values
    
    self._alphas = self._alphas[:, indices]
    self._alphas = self._alphas[:, self._lambdas > 0]  # selecting values only for non zero eigen values
    
    self.X_fit = X

    
  def _transform(self):
    return self.X_fit.dot(self._alphas)
  
  
  def transform(self, X):
    return X.dot(self._alphas)
  
  
  def fit(self, X, Y):
    self._fit(X, Y)
    return
  
  
  def fit_transform(self, X, Y):
    self.fit(X, Y)
    return self._transform()