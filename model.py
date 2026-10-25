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

# Step 4 - predict_tree
def predict_tree(tree, X):
  X = np.asarray(X)
  preds = np.empty(X.shape[0], dtype=int)
  for i in range(X.shape[0]):
    node = tree
    while not node["leaf"]:
      if X[i, node["feature"]] <= node["threshold"]:
        node = node["left"]
      else:
        node = node["right"]
    preds[i] = node["value"]
  return preds

# Step 5 - fit_sklearn_tree
from sklearn.tree import DecisionTreeClassifier


def fit_sklearn_tree(X, y, max_depth=2, min_samples_leaf=1, random_state=42):
  clf = DecisionTreeClassifier(
      criterion="gini",
      max_depth=max_depth,
      min_samples_leaf=min_samples_leaf,
      random_state=random_state,
  )
  clf.fit(X, y)
  return clf

# Step 6 - sklearn_splits
def sklearn_splits(clf):
  tree = clf.tree_
  splits = []
  for i in range(tree.node_count):
    if tree.children_left[i] != -1:  # 非叶子节点
      splits.append((int(tree.feature[i]), round(float(tree.threshold[i]), 3)))
  return splits

# Step 7 - compare_trees
def compare_trees(X, y, max_depth=2):
  my_tree = grow_tree(X, y, max_depth=max_depth, min_samples_leaf=1, depth=0)
  sk_clf = fit_sklearn_tree(
      X, y, max_depth=max_depth, min_samples_leaf=1, random_state=42
  )

  def extract_splits(node):
    if node["leaf"]:
      return []
    res = [(int(node["feature"]), round(float(node["threshold"]), 3))]
    res.extend(extract_splits(node["left"]))
    res.extend(extract_splits(node["right"]))
    return res

  my_splits = extract_splits(my_tree)
  sk_splits = sklearn_splits(sk_clf)
  same_splits = my_splits == sk_splits

  my_preds = predict_tree(my_tree, X)
  sk_preds = sk_clf.predict(X)
  agreement = float(np.mean(my_preds == sk_preds))

  return {
      "same_splits": same_splits,
      "agreement": agreement,
      "my_splits": my_splits,
  }

# Step 8 - moons_data
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def moons_data(n_samples=300, noise=0.25, random_state=42, test_size=0.3):
  X, y = make_moons(
      n_samples=n_samples, noise=noise, random_state=random_state
  )
  return train_test_split(
      X, y, test_size=test_size, random_state=random_state
  )

# Step 9 - overfit_vs_regularized
from sklearn.metrics import accuracy_score


def overfit_vs_regularized(
    X_train, X_test, y_train, y_test, min_samples_leaf=5
):
  free = fit_sklearn_tree(X_train, y_train, max_depth=None, min_samples_leaf=1)
  reg = fit_sklearn_tree(
      X_train, y_train, max_depth=None, min_samples_leaf=min_samples_leaf
  )
  return {
    "free": {
      "train_acc": float(accuracy_score(y_train, free.predict(X_train))),
      "test_acc": float(accuracy_score(y_test, free.predict(X_test))),
      "leaves": int(free.get_n_leaves())
    },
    "regularized": {
      "train_acc": float(accuracy_score(y_train, reg.predict(X_train))),
      "test_acc": float(accuracy_score(y_test, reg.predict(X_test))),
      "leaves": int(reg.get_n_leaves())
    }
  }

# Step 10 - rotation_sensitivity
def rotation_sensitivity(
    X_train, X_test, y_train, y_test, degrees=45.0, min_samples_leaf=5
):
  theta = np.radians(degrees)
  R = np.array(
      [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
  )
  X_train_rot = X_train @ R.T
  X_test_rot = X_test @ R.T

  clf_orig = fit_sklearn_tree(
      X_train, y_train, max_depth=None, min_samples_leaf=min_samples_leaf
  )
  clf_rot = fit_sklearn_tree(
      X_train_rot, y_train, max_depth=None, min_samples_leaf=min_samples_leaf
  )

  orig_acc = float(accuracy_score(y_test, clf_orig.predict(X_test)))
  rot_acc = float(accuracy_score(y_test, clf_rot.predict(X_test_rot)))
  drop = round(float(orig_acc - rot_acc), 4)

  return {"original_acc": orig_acc, "rotated_acc": rot_acc, "drop": drop}

# Step 11 - regression_tree
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor


def regression_tree(X, y, max_depth=2, random_state=42):
  model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
  model.fit(X, y)
  preds = model.predict(X)
  return {
      "model": model,
      "n_distinct_predictions": int(len(np.unique(preds))),
      "train_mse": float(mean_squared_error(y, preds)),
  }

# Step 12 - tree_rules
from sklearn.tree import export_text


def tree_rules(clf, feature_names):
  return export_text(clf, feature_names=list(feature_names))

# Step 13 - save_and_reload_tree
import joblib


def save_and_reload_tree(clf, path):
  joblib.dump(clf, path)
  return joblib.load(path)

# Step 14 - predict_species
def predict_species(clf, measurements, target_names):
  preds = clf.predict(measurements)
  return [str(target_names[i]) for i in preds]

