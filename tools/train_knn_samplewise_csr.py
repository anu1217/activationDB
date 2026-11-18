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
    sample_ids = partial_training_df["run_lbl"].unique().to_list()

    features, sparse_rows = [], []

    # for sample_number, sample_id in enumerate(sample_ids, start=1):
    for sample_id in sample_ids:
        sample_df = partial_training_df.filter(pl.col("run_lbl") == sample_id)
        dense_matrix = np.zeros((len(child_nucs), len(parent_nucs)), dtype=np.float64)
        input_metadata = None

        for row in sample_df.iter_rows(named=True):
            metadata = (row["t_irr"], row["avg_flux_mag"], row["flux_spec_shape_id"])
            if input_metadata is None:
                input_metadata = metadata

            density = row["num_dens_(atoms/cm3)"]

            dense_matrix[
                child_nucs.index(row["nuclide"]),
                parent_nucs.index(row["block_name"])
            ] = density

        t_irr, avg_flux_mag, flux_spec_shape_id = input_metadata

        sample_features = np.asarray(
            [t_irr, avg_flux_mag, *np.asarray(json.loads(dict_df[flux_spec_shape_id]), dtype=np.float64)], dtype=np.float64,
        )

        features.append(sample_features)

        # Convert now, rather than storing dense_matrix in an output list.
        sparse_rows.append(csr_matrix(dense_matrix.reshape(1, -1)))
        del dense_matrix
        if progress:
            print(f"Preparing sample with run_lbl: {sample_id}", flush=True)

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
    parser.add_argument("--output-dir", "-o", default="csr_knn_training_output")
    parser.add_argument("--cv", type=int, default=5, help="GridSearch cross-validation folds (default: 5).")
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
    with open(Path(args.model_training_yaml).expanduser().resolve(), 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    batch_size = config["batch_size"]
    db_path = config['db_name']

    try:
        import extract_sql_data
    except ModuleNotFoundError as error:
        if error.name != "extract_sql_data":
            raise
        raise ModuleNotFoundError(
            "Place the PR's extract_sql_data.py beside train_knn_samplewise_csr.py."
        ) from error

    children, parents = config["child_nucs"], config["parent_nucs"]
    connection = sqlite3.connect(db_path)
    try:
        flux_table = extract_sql_data.make_dict_from_small_table(connection)
        if args.use_stored_df:
            frame = pl.read_csv(
                args.stored_df,
                schema_overrides={
                    "nuclide": pl.Enum(children), 
                    "run_lbl": pl.Categorical,
                    "block_name": pl.Enum(parents), 
                    "flux_spec_shape_id": pl.UInt8,
                },
            )
        else:
            chunks = extract_sql_data.make_dfs_from_num_dens_table(
                connection, config["filter_str"], batch_size, children, parents,
            )
            frame = pl.concat(chunks)
            del chunks
    finally:
        connection.close()
    X_train, Y_csr, sample_ids = make_training_features_outputs_csr(
        frame, flux_table, children, parents, progress=True,
    )
    # The raw SQL DataFrame is no longer needed during cross-validation.
    del frame, flux_table
    target_bytes = int(Y_csr.data.nbytes + Y_csr.indices.nbytes + Y_csr.indptr.nbytes)
    print(
        f"Prepared X={X_train.shape}, CSR targets={Y_csr.shape}, "
        f"nonzeros={Y_csr.nnz:,}, CSR storage={target_bytes / 2**30:.3f} GiB.",
        flush=True,
    )
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
