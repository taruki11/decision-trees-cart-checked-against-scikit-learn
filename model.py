"""
Decision Trees: CART, Checked Against Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - gini
import numpy as np


def gini(y):
  y = np.asarray(y)
  n = len(y)
  if n == 0:
    return 0.0
  _, counts = np.unique(y, return_counts=True)
  p = counts / n
  return float(1.0 - np.sum(p**2))

# Step 2 - best_split
def best_split(X, y):
  X = np.asarray(X)
  y = np.asarray(y)
  n, n_features = X.shape
  if n <= 1:
    return (None, None, 0.0)

  parent_g = gini(y)
  best_gain = 0.0
  best_feature = None
  best_threshold = None

  for j in range(n_features):
    v = X[:, j]
    distinct = np.unique(v)
    if len(distinct) <= 1:
      continue
    # 取连续有序独特值的中点作为候选切分阈值
    thresholds = (distinct[:-1] + distinct[1:]) / 2.0
    for t in thresholds:
      left_mask = v <= t
      n_l = int(np.sum(left_mask))
      n_r = n - n_l
      if n_l == 0 or n_r == 0:
        continue
      g = parent_g - (n_l * gini(y[left_mask]) + n_r * gini(y[~left_mask])) / n
      if g > best_gain:
        best_gain = g
        best_feature = j
        best_threshold = t

  if best_feature is None or best_gain <= 0.0:
    return (None, None, 0.0)
  return (int(best_feature), float(best_threshold), float(best_gain))

# Step 3 - grow_tree
def grow_tree(X, y, max_depth=2, min_samples_leaf=1, depth=0):
  X = np.asarray(X)
  y = np.asarray(y)
  n, n_features = X.shape

  # 多数表决（平局选类别编号较小者）
  classes, counts = np.unique(y, return_counts=True)
  max_c = np.max(counts)
  maj_class = int(np.min(classes[counts == max_c]))

  # 终止条件：纯净节点或达到最大深度
  if len(classes) <= 1 or depth >= max_depth:
    return {"leaf": True, "value": maj_class, "n": n}

  # 寻找满足 min_samples_leaf 的最佳切分
  parent_g = gini(y)
  best_gain = 0.0
  best_feature = None
  best_threshold = None

  for j in range(n_features):
    v = X[:, j]
    distinct = np.unique(v)
    if len(distinct) <= 1:
      continue
    thresholds = (distinct[:-1] + distinct[1:]) / 2.0
    for t in thresholds:
      left_mask = v <= t
      n_l = int(np.sum(left_mask))
      n_r = n - n_l
      if n_l < min_samples_leaf or n_r < min_samples_leaf:
        continue
      g = parent_g - (n_l * gini(y[left_mask]) + n_r * gini(y[~left_mask])) / n
      if g > best_gain:
        best_gain = g
        best_feature = j
        best_threshold = t

  if best_feature is None or best_gain <= 0.0:
    return {"leaf": True, "value": maj_class, "n": n}

  left_mask = X[:, best_feature] <= best_threshold
  return {
      "leaf": False,
      "feature": int(best_feature),
      "threshold": float(best_threshold),
      "n": n,
      "left": grow_tree(
          X[left_mask],
          y[left_mask],
          max_depth=max_depth,
          min_samples_leaf=min_samples_leaf,
          depth=depth + 1,
      ),
      "right": grow_tree(
          X[~left_mask],
          y[~left_mask],
          max_depth=max_depth,
          min_samples_leaf=min_samples_leaf,
          depth=depth + 1,
      ),
  }

# Step 4 - predict_tree (not yet solved)
# TODO: implement

# Step 5 - fit_sklearn_tree (not yet solved)
# TODO: implement

# Step 6 - sklearn_splits (not yet solved)
# TODO: implement

# Step 7 - compare_trees (not yet solved)
# TODO: implement

# Step 8 - moons_data (not yet solved)
# TODO: implement

# Step 9 - overfit_vs_regularized (not yet solved)
# TODO: implement

# Step 10 - rotation_sensitivity (not yet solved)
# TODO: implement

# Step 11 - regression_tree (not yet solved)
# TODO: implement

# Step 12 - tree_rules (not yet solved)
# TODO: implement

# Step 13 - save_and_reload_tree (not yet solved)
# TODO: implement

# Step 14 - predict_species (not yet solved)
# TODO: implement

