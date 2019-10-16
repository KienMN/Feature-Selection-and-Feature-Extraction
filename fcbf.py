"""
Fast Correlation-based Filter algorithm.

Paper: Feature Selection for High-Dimensional Data: A Fast Correlation-Based Filter Solution. Yu & Liu (ICML 2003).

Reference: https://github.com/shiralkarprashant/FCBF
"""

import numpy as np

def entropy(vec, base=2):
  """Returns the empirical entropy H(X) in the input vector."""
  _, vec = np.unique(vec, return_counts=True)
  prob_vec = np.array(vec/float(sum(vec)))
  if base == 2:
    logfn = np.log2
  elif base == 10:
    logfn = np.log10
  else:
    logfn = np.log
  return -prob_vec.dot(logfn(prob_vec))

def conditional_entropy(x, y):
  """Returns H(X|Y)."""
  uy, uyc = np.unique(y, return_counts=True)
  prob_uyc = uyc/float(sum(uyc))
  cond_entropy_x = np.array([entropy(x[y == v]) for v in uy])
  return prob_uyc.dot(cond_entropy_x)

def mutual_information(x, y):
  """Returns the information gain/mutual information [H(X) - H(X|Y)] between two random vars x & y."""
  return entropy(x) - conditional_entropy(x, y)

def symmetrical_uncertainty(x, y):
  """Returns symmetrical uncertainty (SU) - a symmetric mutual information measure."""
  return 2.0 * mutual_information(x, y) / (entropy(x) + entropy(y))

def get_first_element(d):
  """
  Returns tuple corresponding to first 'unconsidered' feature.

  Parameters
  ----------
  d : ndarray
    A 2-d array with SU, original feature index and flag as columns.

  Returns
  -------
  a, b, c: tuple
    a - SU value, b - original feature index, c - index of next 'unconsidered' feature.
  """

  t = np.where(d[:, 2] > 0)[0]
  if len(t):
    return d[t[0], 0], d[t[0], 1], t[0]
  return None, None, None

def get_next_element(d, idx):
  """
  Returns tuple corresponding to the next 'unconsidered' feature.

  Parameters
  ----------
  d : 2-D ndarray
		A 2-d array with SU, original feature index and flag as columns.
	idx : int
		Represents original index of a feature whose next element is required.

  Returns
  -------
  a, b, c : tuple
		a - SU value, b - original feature index, c - index of next 'unconsidered' feature.
  """
  t = np.where(d[:, 2] > 0)[0]
  t = t[t > idx]
  if len(t):
    return d[t[0], 0], d[t[0], 1], t[0]
  return None, None, None

def remove_element(d, idx):
  """
	Returns data with requested feature removed.
	
	Parameters:
	-----------
	d : 2-D ndarray
		A 2-d array with SU, original feature index and flag as columns.
	idx : int
		Represents original index of a feature which needs to be removed.
		
	Returns:
	--------
	d : ndarray
		Same as input, except with specific feature removed.
	"""
  d[idx, 2] = 0
  return d

def c_correlation(X, y):
  """
  Returns SU values between each feature and class.

  Parameters
  ----------
  X : 2-D ndarray
    Feature matrix.

  y : ndarray
    Class label vector.

  Returns
  -------
  su : ndarray
    Symmetric Uncertainty (SU) values for each feature.
  """
  n_features = X.shape[1]
  su = np.zeros(n_features)
  for i in range(n_features):
    su[i] = symmetrical_uncertainty(X[:, i], y)
  return su

def fcbf(X, y, threshold):
  """
  Perform Fast Correlation-Based Filter solution (FCBF).

  Parameters
  ----------
  X : 2-D ndarray
		Feature matrix

	y : ndarray
		Class label vector

	thresh : float
		A value in [0,1) used as threshold for selecting 'relevant' features. 
		A negative value suggest the use of minimum SU[i,c] value as threshold.

  Returns:
	--------
	sbest : 2-D ndarray
		An array containing SU[i,c] values and feature index i.
  """
  n = X.shape[1]
  slist = np.zeros((n, 3))
  slist[:, -1] = 1

  # Identify relevant features
  slist[:, 0] = c_correlation(X, y) # Compute 'C-correlation'
  idx = slist[:, 0].argsort()[::-1]
  slist = slist[idx,]
  slist[:, 1] = idx

  if thresh < 0:
    thresh = np.median(slist[-1, 0])
    print("Using minimum SU value as default threshold: {0}".format(thresh))
  elif thresh >= 1 or thresh > max(slist[:, 0]):
    print("No relevant features selected for given threshold.")
    print("Please lower the threshold and try again.")
    return None

  slist = slist[slist[:, 0] > thresh, :]

  # Identify redundant features among the relevant ones
  cache = {}
  m = len(slist)
  p_su, p, p_idx = get_first_element(slist)

  for _ in range(m):
    p = int(p)
    q_su, q, q_idx = get_next_element(slist, p_idx)
    if q:
      while q:
        q = int(q)
        if (p, q) in cache:
          pq_su = cache[(p, q)]
        else:
          pq_su = symmetrical_uncertainty(X[:, p], X[:, q])
          cache[(p, q)] = pq_su
        
        if pq_su > q_su:
          slist = remove_element(slist, q_idx)
        q_su, q, q_idx = get_next_element(slist, q_idx)
    p_su, p, p_idx = get_next_element(slist, p_idx)
    if not p_idx:
      break
  sbest = slist[slist[:, 2] > 0, :2]
  return sbest