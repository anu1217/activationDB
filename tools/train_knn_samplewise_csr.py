"""Construct one dense sample at a time, convert to CSR, then tune one KNN.

Requires NumPy, SciPy, Polars, scikit-learn, PyYAML, and joblib. Keep
sparse_target_knn.py and the PR's extract_sql_data.py beside this file.
The sparse-aware wrapper is unchanged.

Run complete database extraction, preparation, training, and saving with:

    python train_knn_samplewise_csr.py -t model_training.yaml -o knn_output

The YAML uses the original keys: db_name, filter_str, batch_size, child_nucs,
and parent_nucs. Relative db_name paths are resolved against the YAML file's
directory. Output paths are resolved against the current working directory.
Optional --cv changes the number of CV folds (default 5). To read an already
extracted CSV instead of the number_densities table, pass --use_stored_df
and optionally --stored-df selected_df.csv. The database is still used for
the flux-spectrum lookup table. Raw extraction does not automatically write
a CSV cache.

The output directory contains knn_model.joblib (a model/metadata dictionary),
knn_cv_results.csv, and knn_training_summary.json. CV selects hyperparameters;
the final selected pipeline is refitted on all supplied training samples.

Use after your existing SQL extraction has produced partial_training_df and
dict_df, with ordered child_nucs/parent_nucs from the original configuration:

    from train_knn_samplewise_csr import (
        make_training_features_outputs_csr, make_knn_search,
    )
    X_train, Y_csr, sample_ids = make_training_features_outputs_csr(
        partial_training_df, dict_df, child_nucs, parent_nucs,
    )
    # Y_csr has shape (17000, 533962) for your complete maximum-size dataset.
    search = make_knn_search()
    search.fit(X_train, Y_csr)
    predictions_csr = search.predict(X_new)
    # Convert only manageable prediction batches if dense outputs are wanted:
    predictions_dense = predictions_csr.toarray()

To save the fitted model together with the output-coordinate definitions:

    import joblib
    joblib.dump({
        "model": search.best_estimator_,
        "sample_ids": sample_ids,
        "child_nucs": list(child_nucs),
        "parent_nucs": list(parent_nucs),
        "target_shape": (len(child_nucs), len(parent_nucs)),
        "flatten_order": "C",
    }, "knn_model.joblib")

This replaces the original preparation-function call and Y.reshape line.
It preserves the PR's CHILD x PARENT matrix orientation, then C-flattens it.
To display a dense prediction as PARENT x CHILD, reshape it to
(n_queries, n_children, n_parents) and transpose axes to (0, 2, 1).

Memory: the full dense target dataset is never constructed. Only one sample's
dense output matrix is retained at a time. All compressed target rows remain
in RAM; final vstack needs additional sparse storage. X remains dense and the
already-extracted SQL DataFrame also remains in RAM. This function does not
make SQL extraction or the fitted estimator disk-backed.

As in the original code, absent parent-child records are interpreted as true
zeros, and duplicate sample/parent/child records use the last value. Each
sample must have exactly one input-feature combination. Sample IDs are sorted
so X and Y have deterministic matching order. A sample absent entirely from
the DataFrame cannot be reconstructed by this helper.
"""

import argparse
import csv
import json
from pathlib import Path
import sqlite3

import joblib
import numpy as np
import polars as pl
import sklearn
import yaml
from scipy.sparse import csr_matrix, vstack
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sparse_target_knn import (
    SparseTargetKNNRegressor,
    sparse_neg_mean_squared_error,
)


def make_training_features_outputs_csr(
    partial_training_df, dict_df, child_nucs, parent_nucs, *, progress=False,
):
    """Return aligned dense X, sparse Y, and sample IDs from the PR's rows.

    Sample matrices are constructed exactly as child-by-parent dense arrays,
    immediately converted to one CSR row, and then released. No dense array
    with a sample dimension is created for the targets.
    """
    child_to_index = {name: i for i, name in enumerate(child_nucs)}
    parent_to_index = {name: i for i, name in enumerate(parent_nucs)}
    if not child_to_index or not parent_to_index:
        raise ValueError("child_nucs and parent_nucs must be nonempty.")
    if len(child_to_index) != len(child_nucs) or len(parent_to_index) != len(parent_nucs):
        raise ValueError("child_nucs and parent_nucs must not contain duplicates.")
    sample_ids = sorted(partial_training_df["run_lbl"].unique().to_list())
    if not sample_ids:
        raise ValueError("No samples were found in partial_training_df.")
    features, sparse_rows = [], []
    spectrum_cache = {}

    for sample_number, sample_id in enumerate(sample_ids, start=1):
        sample_df = partial_training_df.filter(pl.col("run_lbl") == sample_id)
        dense_matrix = np.zeros((len(child_nucs), len(parent_nucs)), dtype=np.float64)
        input_metadata = None

        for row in sample_df.iter_rows(named=True):
            metadata = (row["t_irr"], row["avg_flux_mag"], row["flux_spec_shape_id"])
            if input_metadata is None:
                input_metadata = metadata
            elif metadata != input_metadata:
                raise ValueError(f"Sample {sample_id!r} has inconsistent input features.")
            density = float(row["num_dens_(atoms/cm3)"])
            if not np.isfinite(density):
                raise ValueError(f"Sample {sample_id!r} contains a nonfinite target.")
            dense_matrix[
                child_to_index[row["nuclide"]],
                parent_to_index[row["block_name"]],
            ] = density

        t_irr, avg_flux_mag, spectrum_id = input_metadata
        if spectrum_id not in spectrum_cache:
            spectrum = dict_df[spectrum_id]
            if isinstance(spectrum, str):
                spectrum = json.loads(spectrum)
            spectrum = np.asarray(spectrum, dtype=np.float64)
            if spectrum.ndim != 1 or not len(spectrum):
                raise ValueError("Each flux spectrum must be a nonempty 1-D sequence.")
            spectrum_cache[spectrum_id] = spectrum
        sample_features = np.asarray(
            [t_irr, avg_flux_mag, *spectrum_cache[spectrum_id]], dtype=np.float64,
        )
        if not np.isfinite(sample_features).all():
            raise ValueError(f"Sample {sample_id!r} contains nonfinite input features.")
        if features and sample_features.shape != features[0].shape:
            raise ValueError("Flux spectra must use the same number of energy bins.")
        features.append(sample_features)

        # Convert now, rather than storing dense_matrix in an output list.
        sparse_rows.append(csr_matrix(dense_matrix.reshape(1, -1)))
        del dense_matrix
        if progress:
            print(f"\rPreparing samples: {sample_number}/{len(sample_ids)}", end="", flush=True)

    if progress:
        print()
    return np.stack(features), vstack(sparse_rows, format="csr"), sample_ids


def make_knn_search(*, cv=5, param_grid=None, verbose=0):
    """Create the smaller KNN-only grid search, with sparse-aware scoring."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", SparseTargetKNNRegressor(
            prediction_format="csr", prediction_batch_size=16, copy_y=False,
        )),
    ])
    if param_grid is None:
        param_grid = {
            "regressor__n_neighbors": [1, 3, 5, 10, 20],
            "regressor__weights": ["uniform", "distance"],
            "regressor__p": [1, 2],
        }
    return GridSearchCV(
        pipeline, param_grid=param_grid, cv=cv,
        scoring=sparse_neg_mean_squared_error,
        n_jobs=1, pre_dispatch=1, error_score="raise", verbose=verbose,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model_training_yaml", "--model-training-yaml", "-t", required=True,
        help="YAML containing db_name, filter_str, batch_size, child_nucs, and parent_nucs.",
    )
    parser.add_argument("--output-dir", "-o", default="knn_training_output")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds (default: 5).")
    parser.add_argument(
        "--use_stored_df", "--use-stored-df", "-u", action="store_true",
        help="Use an existing extracted CSV; the database still supplies flux spectra.",
    )
    parser.add_argument("--stored-df", default="selected_df.csv")
    return parser.parse_args(argv)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _write_cv_results(search, destination):
    results = search.cv_results_
    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(results))
        writer.writeheader()
        for index in range(len(results["params"])):
            row = {}
            for name, values in results.items():
                value = values[index]
                if np.ma.is_masked(value):
                    value = ""
                elif name == "params":
                    value = json.dumps(value, default=_json_default, sort_keys=True)
                elif isinstance(value, np.generic):
                    value = value.item()
                row[name] = value
            writer.writerow(row)


def main(argv=None):
    """Load configuration/data, prepare CSR targets, train, and save outputs."""
    args = parse_args(argv)
    config_path = Path(args.model_training_yaml).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("The training YAML must contain a mapping of configuration keys.")
    required = {"db_name", "filter_str", "batch_size", "child_nucs", "parent_nucs"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing YAML keys: {', '.join(missing)}")
    if not isinstance(config["db_name"], str) or not config["db_name"]:
        raise ValueError("db_name must be a nonempty database path string.")
    if not isinstance(config["filter_str"], str):
        raise ValueError("filter_str must be an SQL LIKE-pattern string.")
    batch_size = config["batch_size"]
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    for label in ("child_nucs", "parent_nucs"):
        entries = config[label]
        if not isinstance(entries, list) or not entries or not all(isinstance(x, str) for x in entries):
            raise ValueError(f"{label} must be a nonempty list of nuclide-name strings.")
        if len(set(entries)) != len(entries):
            raise ValueError(f"{label} must not contain duplicate nuclide names.")
    if args.cv < 2:
        raise ValueError("--cv must be at least 2.")
    db_path = Path(config["db_name"]).expanduser()
    if not db_path.is_absolute():
        db_path = config_path.parent / db_path
    db_path = db_path.resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"Database not found: {db_path}")

    # Keep SQL loading in the original helper; it must be available beside us
    # or on PYTHONPATH. Imports remain optional for the preparation-only API.
    try:
        import extract_sql_data
    except ModuleNotFoundError as error:
        if error.name != "extract_sql_data":
            raise
        raise ModuleNotFoundError(
            "Place the PR's extract_sql_data.py beside train_knn_samplewise_csr.py."
        ) from error

    children, parents = config["child_nucs"], config["parent_nucs"]
    print(f"Loading training data from {db_path}", flush=True)
    connection = sqlite3.connect(str(db_path))
    try:
        flux_lookup = extract_sql_data.make_dict_from_small_table(connection)
        if args.use_stored_df:
            frame = pl.read_csv(
                Path(args.stored_df).expanduser(),
                schema_overrides={
                    "nuclide": pl.Enum(children), "run_lbl": pl.Categorical,
                    "block_name": pl.Enum(parents), "flux_spec_shape_id": pl.UInt8,
                },
            )
        else:
            chunks = list(extract_sql_data.make_dfs_from_num_dens_table(
                connection, config["filter_str"], batch_size, children, parents,
            ))
            if not chunks:
                raise ValueError("No training rows matched filter_str.")
            frame = pl.concat(chunks)
            del chunks
    finally:
        connection.close()
    print(f"Loaded {frame.height:,} database rows; preparing sample-wise CSR targets.", flush=True)
    X_train, Y_csr, sample_ids = make_training_features_outputs_csr(
        frame, flux_lookup, children, parents, progress=True,
    )
    # The raw SQL DataFrame is no longer needed during cross-validation.
    del frame, flux_lookup
    target_bytes = int(Y_csr.data.nbytes + Y_csr.indices.nbytes + Y_csr.indptr.nbytes)
    print(
        f"Prepared X={X_train.shape}, CSR targets={Y_csr.shape}, "
        f"nonzeros={Y_csr.nnz:,}, CSR storage={target_bytes / 2**30:.3f} GiB.",
        flush=True,
    )
    print("Selecting KNN hyperparameters; final refit will use all training samples.", flush=True)
    search = make_knn_search(cv=args.cv, verbose=2)
    search.fit(X_train, Y_csr)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "knn_model.joblib"
    joblib.dump({
        "model": search.best_estimator_,
        "sample_ids": sample_ids,
        "child_nucs": list(children), "parent_nucs": list(parents),
        "target_shape": (len(children), len(parents)), "flatten_order": "C",
        "best_params": search.best_params_,
        "best_cv_neg_mean_squared_error": float(search.best_score_),
        "sklearn_version": sklearn.__version__,
    }, model_path)
    _write_cv_results(search, output_dir / "knn_cv_results.csv")
    summary = {
        "n_samples": len(sample_ids), "n_features": X_train.shape[1],
        "n_outputs": Y_csr.shape[1], "target_shape": [len(children), len(parents)],
        "target_order": "child,parent", "flatten_order": "C",
        "target_nnz": int(Y_csr.nnz), "target_csr_bytes": target_bytes,
        "cv_folds": args.cv, "best_params": search.best_params_,
        "best_cv_neg_mean_squared_error": float(search.best_score_),
        "best_cv_mean_squared_error": -float(search.best_score_),
        "final_refit_samples": len(sample_ids),
        "sklearn_version": sklearn.__version__,
    }
    with (output_dir / "knn_training_summary.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, default=_json_default)
        output.write("\n")
    print(f"Best settings: {search.best_params_}", flush=True)
    print(f"Best CV MSE: {-search.best_score_:.8g}", flush=True)
    print(f"Saved model: {model_path}", flush=True)
    print(f"Saved CV results and training summary in {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
