import pickle
from typing import List, Union, Dict
import numpy as np

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the element-wise sigmoid function in a numerically stable way.

    Args:
        x: Input array (any shape). Values will be cast to float.

    Returns:
        An ndarray of the same shape as `x` with sigmoid applied element-wise.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out

def load_model(pickle_path: str) -> Dict:
    """
    Load and validate a DLATK model pickle for inference.

    This function opens and unpickles the file at `pickle_path` and performs
    lightweight validation. It does not mutate the returned dictionary.

    Args:
        pickle_path: Path to the DLATK model pickle file.

    Returns:
        The unmodified dictionary loaded from the pickle.

    Raises:
        ValueError: If required keys are missing, if both classification and
                    regression model dicts are present, or if the model uses
                    multiX/controls which this backend does not support.
    """
    with open(pickle_path, "rb") as f:
        model_dict = pickle.load(f)

    if "featureNames" not in model_dict:
        raise ValueError("Missing required key 'featureNames' in model pickle.")
    has_clf = "classificationModels" in model_dict
    has_reg = "regressionModels" in model_dict
    if not (has_clf ^ has_reg):
        raise ValueError("Model pickle must contain either 'classificationModels' or 'regressionModels' (but not both).")

    if model_dict.get("multiXOn"):
        raise ValueError("Model uses multiX (multiXOn=True) which is not supported by this backend.")
    if "controlsOrder" in model_dict:
        raise ValueError("Model contains controls (controlsOrder) which is not supported by this backend.")

    return model_dict

def _align_and_validate_X(
    X: Union[List[List[float]], np.ndarray],
    feature_names: List[str],
    model_feature_names: List[str],
) -> np.ndarray:
    """
    Validate feature name set equality and reorder columns of X to match model order.

    Args:
        X: Input data as a list-of-lists or a numpy array (samples x features).
        feature_names: Column names corresponding to columns of X (order may differ).
        model_feature_names: Feature names expected by the model in the correct order.

    Returns:
        Numpy array of shape (n_samples, n_model_features) where columns are
        reordered to match model_feature_names.

    Raises:
        ValueError: If feature name sets differ or X is not 2-dimensional.
    """
    if set(feature_names) != set(model_feature_names):
        raise ValueError(
            f"Feature names mismatch.\nExpected: {model_feature_names}\nReceived: {feature_names}"
        )
    idxs = [feature_names.index(f) for f in model_feature_names]
    X_np = np.asarray(X, dtype=float)
    if X_np.ndim != 2:
        raise ValueError("X must be 2-dimensional (samples x features).")
    return X_np[:, idxs]

def _normalize_outcomes_arg(outcomes, available):
    """
    Normalize the outcomes argument to a list of outcome names.

    Args:
        outcomes: None, a single outcome name (str), or a list of names.
        available: Iterable of available outcome names (used for default).

    Returns:
        A list of outcome names to run.

    Raises:
        ValueError: If `outcomes` is not None, str, or list.
    """
    if outcomes is None:
        return list(available)
    if isinstance(outcomes, str):
        return [outcomes]
    if isinstance(outcomes, list):
        return outcomes
    raise ValueError("outcomes must be None, a string, or a list of strings.")

def predict_classifier_proba(
    X: Union[List[List[float]], np.ndarray],
    feature_names: List[str],
    pickle_path: str,
    outcomes: Union[List[str], str] = None,
) -> Dict[str, List[List[float]]]:
    """
    Predict class probabilities using a DLATK classification model pickle.

    For binary outcomes this returns n x 1 (positive class probability).
    For multiclass outcomes this returns n x n_classes (full probability vectors).

    Args:
        X: Samples x features as list-of-lists or numpy array.
        feature_names: Names of columns in X (order may differ from model).
        pickle_path: Path to DLATK model pickle containing `classificationModels`.
        outcomes: None (run all), a string (single outcome), or a list of outcome names.

    Returns:
        Dict mapping outcome name -> list-of-lists probabilities (rows correspond to samples).

    Raises:
        ValueError: For missing keys, feature mismatches, unknown outcomes, or
                    when a model does not support probability prediction.

    Notes:
        Applies per-outcome scaler and feature-selector if present in the pickle.

    Examples (Python):
        >>> X = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        >>> feature_names = ["feat1", "feat2", "feat3"]
        >>> model_path = "my_clf_model.pkl"
        >>> probs = predict_classifier_proba(X, feature_names, model_path)
        >>> # probs is a dict: { "outcome_name": [[p1], [p2], ...], ... }
    """
    model = load_model(pickle_path)
    if "classificationModels" not in model:
        raise ValueError("Provided pickle does not contain classificationModels.")

    X_aligned = _align_and_validate_X(X, feature_names, model["featureNames"])
    all_outcomes = list(model["classificationModels"].keys())
    outcomes_to_run = _normalize_outcomes_arg(outcomes, all_outcomes)

    missing = [o for o in outcomes_to_run if o not in all_outcomes]
    if missing:
        raise ValueError(f"Outcome(s) not found in model: {missing}")

    predictions_dict: Dict[str, List[List[float]]] = {}
    for out in outcomes_to_run:
        clf = model["classificationModels"][out]
        scaler = model.get("multiScalers", {}).get(out) or None
        fselector = model.get("multiFSelectors", {}).get(out) or None

        X_proc = X_aligned
        if scaler is not None:
            X_proc = scaler.transform(X_proc)
        if fselector is not None:
            X_proc = fselector.transform(X_proc)

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_proc)
        elif hasattr(clf, "decision_function"):
            decision = np.asarray(clf.decision_function(X_proc))
            # binary decision_function -> 1d or (n,1)
            if decision.ndim == 1 or (decision.ndim == 2 and decision.shape[1] == 1):
                if decision.ndim == 2:
                    decision = decision.ravel()
                p_pos = _sigmoid(decision)
                probs = np.vstack([1 - p_pos, p_pos]).T
            else:
                expd = np.exp(decision - np.max(decision, axis=1, keepdims=True))
                probs = expd / expd.sum(axis=1, keepdims=True)
        else:
            raise ValueError(f"Model for outcome '{out}' does not support probability prediction.")

        # For binary, return only positive class probability as single-column output
        if probs.shape[1] == 2:
            result = probs[:, 1:2]
        else:
            result = probs

        predictions_dict[out] = result.tolist()

    return predictions_dict

def predict_classifier_classes(
    X: Union[List[List[float]], np.ndarray],
    feature_names: List[str],
    pickle_path: str,
    outcomes: Union[List[str], str] = None,
) -> Dict[str, List[Union[int, str]]]:
    """
    Predict hard class labels using a DLATK classification model pickle.

    Args:
        X: Samples x features as list-of-lists or numpy array.
        feature_names: Names of columns in X (order may differ from model).
        pickle_path: Path to DLATK model pickle containing `classificationModels`.
        outcomes: None (run all), a string (single outcome), or a list of outcome names.

    Returns:
        Dict mapping outcome name -> list of predicted class labels.

    Raises:
        ValueError: For missing keys, feature mismatches, unknown outcomes, or
                    when a model does not support `.predict()`.

    Notes:
        Applies per-outcome scaler and feature-selector if present in the pickle.

    Examples (Python):
        >>> X = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        >>> feature_names = ["feat1", "feat2", "feat3"]
        >>> classes = predict_classifier_classes(X, feature_names, "my_clf_model.pkl")
        >>> # classes: { "outcome_name": ["pos","neg", ...], ... }
    """
    model = load_model(pickle_path)
    if "classificationModels" not in model:
        raise ValueError("Provided pickle does not contain classificationModels.")

    X_aligned = _align_and_validate_X(X, feature_names, model["featureNames"])
    all_outcomes = list(model["classificationModels"].keys())
    outcomes_to_run = _normalize_outcomes_arg(outcomes, all_outcomes)

    missing = [o for o in outcomes_to_run if o not in all_outcomes]
    if missing:
        raise ValueError(f"Outcome(s) not found in model: {missing}")

    predictions_dict: Dict[str, List[Union[int, str]]] = {}
    for out in outcomes_to_run:
        clf = model["classificationModels"][out]
        scaler = model.get("multiScalers", {}).get(out) or None
        fselector = model.get("multiFSelectors", {}).get(out) or None

        X_proc = X_aligned
        if scaler is not None:
            X_proc = scaler.transform(X_proc)
        if fselector is not None:
            X_proc = fselector.transform(X_proc)

        if not hasattr(clf, "predict"):
            raise ValueError(f"Model for outcome '{out}' does not support class prediction.")

        preds = clf.predict(X_proc)
        predictions_dict[out] = np.asarray(preds).tolist()

    return predictions_dict

def predict_regression_values(
    X: Union[List[List[float]], np.ndarray],
    feature_names: List[str],
    pickle_path: str,
    outcomes: Union[List[str], str] = None,
) -> Dict[str, List[List[float]]]:
    """
    Predict regression values using a DLATK regression model pickle.

    For single-target regressors returns n x 1 lists. Multi-output regressors
    will return n x k lists.

    Args:
        X: Samples x features as list-of-lists or numpy array.
        feature_names: Names of columns in X (order may differ from model).
        pickle_path: Path to DLATK model pickle containing `regressionModels`.
        outcomes: None (run all), a string (single outcome), or a list of outcome names.

    Returns:
        Dict mapping outcome name -> list-of-lists of regression predictions.

    Raises:
        ValueError: For missing keys, feature mismatches, unknown outcomes, or
                    when a model does not support `.predict()`.

    Notes:
        Applies per-outcome scaler and feature-selector if present in the pickle.

    Examples (Python):
        >>> X = [[1.0, 2.0], [3.0, 4.0]]
        >>> feature_names = ["f1","f2"]
        >>> res = predict_regression_values(X, feature_names, "my_reg_model.pkl")
        >>> # res: { "outcome": [[0.5], [1.2]], ... }
    """
    model = load_model(pickle_path)
    if "regressionModels" not in model:
        raise ValueError("Provided pickle does not contain regressionModels.")

    X_aligned = _align_and_validate_X(X, feature_names, model["featureNames"])
    all_outcomes = list(model["regressionModels"].keys())
    outcomes_to_run = _normalize_outcomes_arg(outcomes, all_outcomes)

    missing = [o for o in outcomes_to_run if o not in all_outcomes]
    if missing:
        raise ValueError(f"Outcome(s) not found in model: {missing}")

    predictions_dict: Dict[str, List[List[float]]] = {}
    for out in outcomes_to_run:
        reg = model["regressionModels"][out]
        scaler = model.get("multiScalers", {}).get(out) or None
        fselector = model.get("multiFSelectors", {}).get(out) or None

        X_proc = X_aligned
        if scaler is not None:
            X_proc = scaler.transform(X_proc)
        if fselector is not None:
            X_proc = fselector.transform(X_proc)

        if not hasattr(reg, "predict"):
            raise ValueError(f"Model for outcome '{out}' does not support .predict().")

        preds = np.asarray(reg.predict(X_proc), dtype=float)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        predictions_dict[out] = preds.tolist()

    return predictions_dict

def load_reducer_model(pickle_path: str) -> Dict:
    """
    Load and validate a DLATK dimension-reducer pickle for inference.

    Expected keys in the pickle:
      - "clusterModels" : dict of reducer objects keyed by outcome (or "noOutcome")
      - "scalers"       : dict of scalers (may be empty)
      - "fSelectors"    : dict of feature selectors (may be empty)
      - "featureNames"  : list of feature names in model order

    This function does not mutate the loaded dict.

    Args:
        pickle_path: Path to the reducer pickle file.

    Returns:
        The unmodified dict loaded from the pickle.

    Raises:
        ValueError: If required keys are missing or if the model uses multiX/controls.
    """
    with open(pickle_path, "rb") as f:
        model_dict = pickle.load(f)

    required = ["clusterModels", "featureNames"]
    for k in required:
        if k not in model_dict:
            raise ValueError(f"Missing required key '{k}' in reducer pickle.")

    # Reject unsupported multiX/controls
    if model_dict.get("multiXOn"):
        raise ValueError("Reducer model uses multiX (multiXOn=True) which is not supported by this backend.")
    if "controlsOrder" in model_dict:
        raise ValueError("Reducer model contains controls (controlsOrder) which is not supported by this backend.")

    return model_dict


def apply_dimension_reducer(
    X: Union[List[List[float]], np.ndarray],
    feature_names: List[str],
    pickle_path: str,
    outcomes: Union[List[str], str] = None,
) -> Dict[str, Dict]:
    """
    Apply a saved DLATK dimension reducer to input data.

    Behavior / assumptions:
      - Dense input only (list-of-lists or numpy array).
      - feature_names must match the model's featureNames as a set; order may differ.
      - For each requested outcome (or all), the function:
          * reorders X to model order,
          * applies scaler.transform and fSelector.transform if present,
          * calls cluster.transform(X_proc) if available, else cluster.predict(X_proc).
          * If returned array is 1D, it is reshaped to (n_samples, 1).
      - Component names are derived from cluster.components_ or cluster.n_components_ when available,
        otherwise generated as COMPONENT_0..COMPONENT_{k-1}.

    Args:
        X: samples x features as list-of-lists or numpy array.
        feature_names: names of columns in X (order may differ).
        pickle_path: path to reducer pickle containing 'clusterModels'.
        outcomes: None (run all), a string (single outcome), or a list of outcome names.

    Returns:
        Dict mapping outcome name -> dict with keys:
            - "components": list-of-lists (n_samples x n_components)
            - "component_names": list of str

    Raises:
        ValueError: For feature mismatches, unknown outcomes, unsupported models, or if reducer
                    returns unexpected shapes.
    """
    model = load_reducer_model(pickle_path)
    model_feature_names = model["featureNames"]
    X_aligned = _align_and_validate_X(X, feature_names, model_feature_names)

    all_outcomes = list(model["clusterModels"].keys())
    outcomes_to_run = _normalize_outcomes_arg(outcomes, all_outcomes)

    missing = [o for o in outcomes_to_run if o not in all_outcomes]
    if missing:
        raise ValueError(f"Outcome(s) not found in reducer model: {missing}")

    results: Dict[str, Dict] = {}
    for out in outcomes_to_run:
        cluster = model["clusterModels"][out]
        scaler = model.get("scalers", {}).get(out) or None
        fselector = model.get("fSelectors", {}).get(out) or None

        X_proc = X_aligned
        if scaler is not None:
            X_proc = scaler.transform(X_proc)
        if fselector is not None:
            X_proc = fselector.transform(X_proc)

        # Prefer transform, but fall back to predict if transform not present.
        if hasattr(cluster, "transform"):
            comp = cluster.transform(X_proc)
        elif hasattr(cluster, "predict"):
            comp = cluster.predict(X_proc)
        else:
            raise ValueError(f"Reducer for outcome '{out}' has no transform/predict methods for inference.")

        comp = np.asarray(comp)
        if comp.ndim == 1:
            comp = comp.reshape(-1, 1)

        # Determine number of components and names
        n_comp = None
        comp_names: List[str] = []
        try:
            # cluster.components_ shaped (n_components, n_features) for many reducers
            if hasattr(cluster, "components_"):
                n_comp = int(getattr(cluster, "components_").shape[0])
        except Exception:
            n_comp = None

        if n_comp is None and hasattr(cluster, "n_components_"):
            try:
                n_comp = int(getattr(cluster, "n_components_"))
            except Exception:
                n_comp = None

        if n_comp is None:
            # fallback to produced array width
            n_comp = comp.shape[1] if comp.ndim == 2 else 1

        comp_names = [f"COMPONENT_{i}" for i in range(n_comp)]

        # If comp has more columns than n_comp inferred, adjust names to match actual columns
        if comp.shape[1] != n_comp:
            comp_names = [f"COMPONENT_{i}" for i in range(comp.shape[1])]

        results[out] = {"components": comp.tolist(), "component_names": comp_names}

    return results

