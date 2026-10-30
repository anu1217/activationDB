"""Sparse-target multioutput KNN for scikit-learn pipelines and grid searches.

Requirements: Python >= 3.10, NumPy, SciPy, scikit-learn >= 1.6.
Validated with scikit-learn 1.8.0; see the accompanying usage example below.

Every row of X and y refers to the same training sample. y has one column
per flattened parent/child coordinate. All parents share one neighbor index.
True zero targets may be implicit; missing/unknown targets must be handled
before fitting. This class does not implement masked-target regression.

Example (run after preparing your arrays):

    from scipy.sparse import load_npz
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV, KFold
    from sparse_target_knn import (
        SparseTargetKNNRegressor, sparse_neg_mean_squared_error,
    )

    Y = load_npz("training_targets.npz")  # (17000, 533962), CSR
    # X: (17000, G + 2), in the SAME sample order as Y.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", SparseTargetKNNRegressor(
            prediction_format="csr", prediction_batch_size=16, copy_y=False,
        )),
    ])
    search = GridSearchCV(
        pipeline,
        param_grid={
            "regressor__n_neighbors": [1, 3, 5, 10, 20],
            "regressor__weights": ["uniform", "distance"],
            "regressor__p": [1, 2],
        },
        cv=KFold(n_splits=5, shuffle=True, random_state=1),
        scoring=sparse_neg_mean_squared_error,
        n_jobs=1, pre_dispatch=1, error_score="raise",
    )
    search.fit(X, Y)  # CV uses train folds; final refit uses ALL samples.
    prediction_csr = search.best_estimator_.predict(X_new)
    # Only convert manageable batches to dense:
    prediction_dense = prediction_csr.toarray()

The scorer is negative MSE averaged over ALL samples and output coordinates,
including implicit zeros. score() uses that same definition, NOT R-squared.
Do not use scoring="neg_mean_squared_error" for CSR predictions; use the
provided callable. Arbitrary sklearn metrics may not accept sparse targets.
Use prediction_format="dense" for consumers requiring NumPy predictions.

Memory: targets are stored in RAM as CSR; this is not a disk-backed estimator.
copy_y=True copies target storage. Float32 targets are promoted to float64.
GridSearchCV creates CSR fold subsets; multiple workers multiply memory use.
Prediction supports internal query batches, but predict() still returns the
complete result. Use iter_predict() to consume or save batches incrementally.
Sparse predictions can have more nonzeros than any individual neighbor.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    check_is_fitted,
    validate_data,
)

__all__ = [
    "SparseTargetKNNRegressor",
    "sparse_neg_mean_squared_error",
]


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")


def _targets_csr(y, *, copy=False):
    """Validate numerical 2-D targets without converting sparse data to dense."""
    validated = check_array(
        y, accept_sparse="csr", dtype=np.float64, ensure_2d=True,
        copy=copy, input_name="y",
    )
    return sparse.csr_matrix(validated, copy=False)


def _row_slice(data, start, stop):
    # Preserve DataFrame feature names in pipelines/scorers.
    if hasattr(data, "iloc"):
        return data.iloc[start:stop]
    if not hasattr(data, "__getitem__"):
        return np.asarray(data)[start:stop]
    return data[start:stop]


def _scorer_batch_size(estimator):
    final_estimator = estimator
    if hasattr(estimator, "steps") and estimator.steps:
        final_estimator = estimator.steps[-1][1]
    batch_size = getattr(final_estimator, "prediction_batch_size", 32)
    _positive_integer(batch_size, "prediction_batch_size")
    return int(batch_size)


def sparse_neg_mean_squared_error(estimator, X, y):
    """GridSearchCV scoring callable: negative MSE, including implicit zeros.

    Works on this estimator or a Pipeline ending in it, with CSR or dense
    predictions. Both validation queries and error calculations are batched.
    No complete dense target or prediction array is constructed here.
    """
    truth = _targets_csr(y)
    check_consistent_length(X, truth)
    n_samples, n_outputs = truth.shape
    squared_error = np.float64(0.0)
    batch_size = _scorer_batch_size(estimator)
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        predicted = _targets_csr(estimator.predict(_row_slice(X, start, stop)))
        expected = truth[start:stop]
        if predicted.shape != expected.shape:
            raise ValueError(
                f"Prediction shape {predicted.shape} does not match "
                f"target shape {expected.shape}."
            )
        error = predicted - expected
        squared_error += error.multiply(error).sum(dtype=np.float64)
    return -float(squared_error / (n_samples * n_outputs))


class SparseTargetKNNRegressor(RegressorMixin, BaseEstimator):
    """KNN regression with one shared input index and sparse multioutput y.

    Parameters mirror NearestNeighbors except weights and output controls.
    weights supports 'uniform' or 'distance'. prediction_format is 'csr'
    (default) or 'dense'. All targets and averages use float64.

    copy_y=False avoids an extra CSR copy when possible; canonicalizing the
    targets may then remove explicit zeros/sum duplicates IN PLACE. Do not
    subsequently mutate the training targets. copy_y=True isolates the fit.

    Targets must be a 2-D array/matrix, including for a single output column.
    predict() always returns a 2-D result. n_neighbors must not exceed the
    number of available training samples when requesting predictions.

    Learned attributes: y_ (CSR training targets), neighbor_index_,
    n_features_in_, n_outputs_, and n_samples_fit_.
    """

    def __init__(
        self,
        n_neighbors=5,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
        prediction_format="csr",
        prediction_batch_size=32,
        copy_y=True,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.prediction_format = prediction_format
        self.prediction_batch_size = prediction_batch_size
        self.copy_y = copy_y

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.input_tags.pairwise = self.metric == "precomputed"
        tags.target_tags.multi_output = True
        tags.target_tags.single_output = False
        tags.target_tags.two_d_labels = True
        tags.regressor_tags.poor_score = True
        return tags

    def _check_parameters(self):
        _positive_integer(self.n_neighbors, "n_neighbors")
        _positive_integer(self.prediction_batch_size, "prediction_batch_size")
        if self.weights not in ("uniform", "distance"):
            raise ValueError("weights must be 'uniform' or 'distance'.")
        self._output_format(None)
        if not isinstance(self.copy_y, (bool, np.bool_)):
            raise ValueError("copy_y must be a boolean.")

    def _output_format(self, override):
        chosen = self.prediction_format if override is None else override
        if chosen not in ("csr", "dense"):
            raise ValueError("output_format/prediction_format must be 'csr' or 'dense'.")
        return chosen

    def fit(self, X, y):
        """Fit using all samples; keep numerical targets in CSR format."""
        self._check_parameters()
        if y is None:
            raise ValueError("This estimator requires y to be passed, but the target y is None.")
        X_validated = validate_data(
            self, X, reset=True, accept_sparse="csr", dtype=np.float64,
        )
        targets = _targets_csr(y, copy=self.copy_y)
        check_consistent_length(X_validated, targets)
        targets.sum_duplicates()
        targets.eliminate_zeros()
        targets.sort_indices()
        neighbor_index = NearestNeighbors(
            n_neighbors=self.n_neighbors, algorithm=self.algorithm,
            leaf_size=self.leaf_size, p=self.p, metric=self.metric,
            metric_params=self.metric_params, n_jobs=self.n_jobs,
        ).fit(X_validated)
        self.y_ = targets
        self.neighbor_index_ = neighbor_index
        self.n_outputs_ = targets.shape[1]
        self.n_samples_fit_ = targets.shape[0]
        return self

    def _validated_queries(self, X):
        check_is_fitted(self, ["neighbor_index_", "y_", "n_outputs_"])
        self._check_parameters()
        return validate_data(
            self, X, reset=False, accept_sparse="csr", dtype=np.float64,
        )

    def _predict_csr_batch(self, X_batch):
        distances, indices = self.neighbor_index_.kneighbors(
            X_batch, n_neighbors=self.n_neighbors, return_distance=True,
        )
        if self.weights == "uniform":
            weights = np.full(distances.shape, 1.0 / self.n_neighbors, dtype=np.float64)
        else:
            if not np.isfinite(distances).all():
                raise ValueError("Neighbor distances must be finite for distance weighting.")
            if (distances < 0).any():
                raise ValueError("Neighbor distances must be nonnegative.")
            zero_distance = distances == 0
            # min_distance/distance gives normalized inverse-distance weights
            # after the row normalization, without overflowing 1/distance.
            minimum = distances.min(axis=1, keepdims=True)
            weights = np.divide(
                minimum, distances, out=np.zeros_like(distances),
                where=~zero_distance,
            )
            exact_match = zero_distance.any(axis=1)
            weights[exact_match] = zero_distance[exact_match]
            weights /= weights.sum(axis=1, keepdims=True)
        n_queries, k = indices.shape
        row_pointers = np.arange(0, (n_queries + 1) * k, k, dtype=np.int64)
        interpolation = sparse.csr_matrix(
            (weights.ravel(), indices.ravel(), row_pointers),
            shape=(n_queries, self.n_samples_fit_),
        )
        interpolation.eliminate_zeros()
        predicted = (interpolation @ self.y_).tocsr()
        predicted.eliminate_zeros()
        return predicted

    def iter_predict(self, X, *, output_format=None):
        """Yield prediction batches in query order; never collect all results."""
        X_validated = self._validated_queries(X)
        chosen = self._output_format(output_format)
        for start in range(0, X_validated.shape[0], self.prediction_batch_size):
            predicted = self._predict_csr_batch(
                X_validated[start:start + self.prediction_batch_size]
            )
            yield predicted if chosen == "csr" else predicted.toarray()

    def predict(self, X, *, output_format=None):
        """Return all predictions as CSR or a dense 2-D NumPy array.

        Internal query batching bounds intermediate calculations, not the
        memory needed for the complete returned result. For bounded output
        memory, consume iter_predict() instead.
        """
        chosen = self._output_format(output_format)
        batches = self.iter_predict(X, output_format="csr")
        if chosen == "csr":
            return sparse.vstack(list(batches), format="csr")
        # Validate through the generator first, then allocate the final result.
        first = next(batches)
        n_queries = X.shape[0] if hasattr(X, "shape") else np.asarray(X).shape[0]
        output = np.empty((n_queries, self.n_outputs_), dtype=np.float64)
        stop = first.shape[0]
        output[:stop] = first.toarray()
        for block in batches:
            start, stop = stop, stop + block.shape[0]
            output[start:stop] = block.toarray()
        return output

    def score(self, X, y):
        """Return negative MSE over all coordinates (not R-squared)."""
        return sparse_neg_mean_squared_error(self, X, y)


if __name__ == "__main__":
    # Small, reproducible smoke example; replace with real arrays for training.
    from sklearn.metrics import mean_squared_error
    from sklearn.neighbors import KNeighborsRegressor

    rng = np.random.default_rng(106)
    X_demo = rng.normal(size=(60, 6))
    Y_demo = rng.random(size=(60, 40))
    Y_demo[rng.random(Y_demo.shape) < 0.92] = 0.0
    X_query = rng.normal(size=(5, 6))
    sparse_model = SparseTargetKNNRegressor(n_neighbors=5, weights="distance")
    sparse_model.fit(X_demo, sparse.csr_matrix(Y_demo))
    actual = sparse_model.predict(X_query, output_format="dense")
    reference = KNeighborsRegressor(n_neighbors=5, weights="distance").fit(
        X_demo, Y_demo,
    ).predict(X_query)
    np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)
    print("Sparse-target predictions match dense KNN.")
    print("Maximum absolute difference:", np.max(np.abs(actual - reference)))
    print("Example MSE:", mean_squared_error(reference, actual))
