import numpy as np
import matplotlib.pyplot as plt
import yaml
from scipy.stats import ecdf

def open_yaml():
    with open("prep_sql_for_ml_model.yaml", 'r') as yaml_file:
        inputs = yaml.safe_load(yaml_file)
    return inputs

def store_parent_errors(knn_file, parent_nuc, filename):
    available_parent_nucs = []
    parent_rmse = []
    parent_mae = []
    parent_r2 = []

    with np.load(filename) as data:
        available_parent_nucs.append(parent_nuc)
        rmse = data["test_rmse"].item()
        mae = data["test_mae"].item()
        r2 = data["test_r2"].item()
        Y_test = data["Y_test"]
        Y_pred = data["Y_pred"]

        parent_rmse.append(rmse)
        parent_mae.append(mae)
        parent_r2.append(r2)

        print(f"{parent_nuc} RMSE: {rmse}", file=knn_file)
        print(f"{parent_nuc} MAE: {mae}", file=knn_file)
        print(f"{parent_nuc} R2: {r2}", file=knn_file)
    return available_parent_nucs, parent_rmse, parent_mae, parent_r2, Y_test, Y_pred

def make_masks(true_densities, abs_delta_N, delta_N):
    del_N_v_N_log_log_mask= ((true_densities > 0) & (abs_delta_N > 0) & (np.isfinite(true_densities)) & (np.isfinite(abs_delta_N)))
    del_N_by_N_v_N_lin_mask = ((true_densities != 0) & np.isfinite(true_densities) & (np.isfinite(delta_N)))

    return del_N_v_N_log_log_mask, del_N_by_N_v_N_lin_mask

def main():
    inputs = open_yaml()
    parent_nucs = inputs["parent_nucs"]
    plot_folder = "knn_joblib_npz_pre_approximation_plots"
    available_parent_nucs = []
    parent_rmse = []
    parent_mae = []
    parent_r2 = []

    with open('knn_metrics_pre_approximation_testing.txt', 'w') as knn_file:
        for parent_nuc in parent_nucs:
            filename = f"knn_joblib_npz_pre_approximation/{parent_nuc}_knn_test_results.npz"

            current_parent_nucs, current_parent_rmse, current_parent_mae, current_parent_r2, Y_test, Y_pred = store_parent_errors(knn_file, parent_nuc, filename)
            available_parent_nucs.extend(current_parent_nucs)
            parent_rmse.extend(current_parent_rmse)
            parent_mae.extend(current_parent_mae)
            parent_r2.extend(current_parent_r2)
            true_densities = Y_test.ravel()
            delta_N = (Y_pred - Y_test).ravel()
            abs_delta_N = np.abs(Y_pred - Y_test).ravel()

            del_N_v_N_log_log_mask, del_N_by_N_v_N_lin_mask = make_masks(true_densities, abs_delta_N, delta_N)
            print(
            f"{parent_nuc} absolute error: {np.sum(del_N_v_N_log_log_mask)} plotted, {np.sum(~del_N_v_N_log_log_mask)} excluded",
            file=knn_file
                )
            if np.any(del_N_v_N_log_log_mask):
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.scatter(true_densities[del_N_v_N_log_log_mask], abs_delta_N[del_N_v_N_log_log_mask], s=5, alpha=0.3)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("True Number Density (atoms/cm³)")
                ax.set_ylabel(r"Absolute Error $\Delta$N (atoms/cm³)")
                ax.set_title(f"Absolute KNN Prediction Error vs True Number Density (Log-Log): {parent_nuc}")
                fig.tight_layout()
                fig.savefig(f"{plot_folder}/{parent_nuc}_abs_delta_N_vs_N_log.png", dpi=300)
                plt.close(fig)

            fractional_errors = np.full(true_densities.shape, np.nan)
            fractional_errors[del_N_by_N_v_N_lin_mask] = (
                delta_N[del_N_by_N_v_N_lin_mask] / true_densities[del_N_by_N_v_N_lin_mask]
            )

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(
                true_densities[del_N_by_N_v_N_lin_mask],
                fractional_errors[del_N_by_N_v_N_lin_mask],
                s=5,
                alpha=0.3
            )
            ax.axhline(0, color="black", linewidth=1)
            ax.set_xlabel("True Number Density (atoms/cm³)")
            ax.set_ylabel("Signed Fractional Error")
            ax.set_title(f"KNN Signed Fractional Error vs True Number Density: {parent_nuc}")
            fig.tight_layout()
            fig.savefig(f"{plot_folder}/{parent_nuc}_fractional_error_linear.png", dpi=300)
            plt.close(fig)

            abs_fractional_errors = np.full(true_densities.shape, np.nan)
            abs_fractional_errors[del_N_by_N_v_N_lin_mask] = (
                np.abs(delta_N[del_N_by_N_v_N_lin_mask]) / true_densities[del_N_by_N_v_N_lin_mask]
            )
            print(f"{parent_nuc} true density min: {np.min(true_densities)}", file=knn_file)
            print(f"{parent_nuc} true density max: {np.max(true_densities)}", file=knn_file)
            print(f"{parent_nuc} delta_N min: {np.min(delta_N)}", file=knn_file)
            print(f"{parent_nuc} delta_N max: {np.max(delta_N)}", file=knn_file)
            print(f"{parent_nuc} positive true densities: {np.sum(true_densities > 0)}", file=knn_file)
            print(f"{parent_nuc} nonzero errors: {np.sum(abs_delta_N > 0)}", file=knn_file)
            print(f"{parent_nuc} finite fractional errors: {np.sum(np.isfinite(abs_fractional_errors))}", file=knn_file)
            print(f"{parent_nuc} positive fractional errors: {np.sum(abs_fractional_errors > 0)}", file=knn_file)            

            del_N_by_N_v_N_log_log_mask = (
                (true_densities > 0)
                & (abs_fractional_errors > 0)
                & (np.isfinite(true_densities))
                & (np.isfinite(abs_fractional_errors))
            )
            print(
                f"{parent_nuc} log fractional error: {np.sum(del_N_by_N_v_N_log_log_mask)} plotted, "
                f"{np.sum(~del_N_by_N_v_N_log_log_mask)} excluded",
                file=knn_file
            )

            if np.any(del_N_by_N_v_N_log_log_mask):
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.scatter(
                    true_densities[del_N_by_N_v_N_log_log_mask],
                    abs_fractional_errors[del_N_by_N_v_N_log_log_mask],
                    s=5,
                    alpha=0.3
                )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("True Number Density (atoms/cm³)")
                ax.set_ylabel("Absolute Fractional Error")
                ax.set_title(
                    f"KNN Absolute Fractional Error vs True Number Density (Log-Log): {parent_nuc}"
                )
                fig.tight_layout()
                fig.savefig(f"{plot_folder}/{parent_nuc}_fractional_error_log.png", dpi=300)
                plt.close(fig)

            # --------------------------- cumulative fraction ecdf begins    

            finite_abs_delta_N = abs_delta_N[np.isfinite(abs_delta_N)]

            if len(finite_abs_delta_N) == 0:
                print(f"{parent_nuc} : No valid absolute errors to plot.", file=knn_file)
            else:
                abs_delta_N_result = ecdf(finite_abs_delta_N)

                plt.figure()
                plt.step(
                    abs_delta_N_result.cdf.quantiles,
                    abs_delta_N_result.cdf.probabilities,
                    where="post"
                )
                plt.xlabel("Absolute Error |Predicted Density - True Density|")
                plt.ylabel("Cumulative Fraction")
                plt.title("ECDF of Absolute Errors")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"{plot_folder}/{parent_nuc}_unmasked_absolute_delta_N_ecdf.png", dpi=300, bbox_inches="tight")
                plt.show()
                plt.close()
            delta_N_mask = ((abs_delta_N > 0) & np.isfinite(abs_delta_N))
            abs_delta_N_masked = abs_delta_N[delta_N_mask]

            print("Total errors:", len(abs_delta_N), file=knn_file)
            print("Finite errors:", np.sum(np.isfinite(abs_delta_N)), file=knn_file)
            print("Positive errors:", np.sum(abs_delta_N > 0), file=knn_file)
            print("Zero errors:", np.sum(abs_delta_N == 0), file=knn_file)
            print("NaN errors:", np.sum(np.isnan(abs_delta_N)), file=knn_file)
            print("Infinite errors:", np.sum(np.isinf(abs_delta_N)), file=knn_file)

            if len(abs_delta_N_masked) == 0:
                print(f"{parent_nuc}: no positive absolute errors; skipping masked ECDF.", file=knn_file)
            else:
                abs_delta_N_result_masked = ecdf(abs_delta_N_masked)

                plt.figure()
                plt.step(
                    abs_delta_N_result_masked.cdf.quantiles,
                    abs_delta_N_result_masked.cdf.probabilities,
                    where="post"
                )
                plt.xscale("log")
                plt.yscale("log")
                plt.xlabel("Absolute Error |Predicted Density - True Density|")
                plt.ylabel("Cumulative Fraction")
                plt.title("Masked ECDF of Absolute Errors")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"{plot_folder}/{parent_nuc}_masked_absolute_delta_N_ecdf.png", dpi=300, bbox_inches="tight")
                plt.show()
                plt.close()
            
            frac_delta_N_masked = abs_delta_N[del_N_v_N_log_log_mask] / true_densities[del_N_v_N_log_log_mask]

            if len(frac_delta_N_masked) == 0:
                print(f"{parent_nuc}: No valid masked fractional errors to plot.", file=knn_file)
            else:

                frac_delta_N_result_masked = ecdf(frac_delta_N_masked)

                plt.figure()
                plt.step(
                    frac_delta_N_result_masked.cdf.quantiles,
                    frac_delta_N_result_masked.cdf.probabilities,
                    where="post"
                )
                plt.xscale("log")
                plt.yscale("log")
                plt.xlabel("Fractional Error |Predicted Density - True Density|")
                plt.ylabel("Cumulative Fraction")
                plt.title("Masked ECDF of Fractional Errors")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"{plot_folder}/{parent_nuc}_masked_fractional_delta_N_ecdf.png", dpi=300, bbox_inches="tight")
                plt.show()
                plt.close()


            # ------------------------------ cumulative fraction ecdf ends

            # ----------------------------- number density-weighted ecdf begins
            if np.any(del_N_v_N_log_log_mask):
                sort_abs_delta_N_indices = np.argsort(
                    abs_delta_N[del_N_v_N_log_log_mask]
                )
                sorted_errors = abs_delta_N[del_N_v_N_log_log_mask][sort_abs_delta_N_indices]
                sorted_densities = true_densities[del_N_v_N_log_log_mask][sort_abs_delta_N_indices]

                cumulative_density = np.cumsum(sorted_densities)
                cumulative_fraction = cumulative_density / np.sum(sorted_densities)

                plt.figure()
                plt.step(sorted_errors, cumulative_fraction, where="post")
                plt.xscale("log")
                plt.yscale("log")
                plt.xlabel("Absolute Error |Predicted Density - True Density|")
                plt.ylabel("Cumulative Fraction of Total Number Density")
                plt.title(f"Number-Density-Weighted ECDF: {parent_nuc}")
                plt.grid(True, which="both")
                plt.tight_layout()
                plt.savefig(
                    f"{plot_folder}/{parent_nuc}_abs_delta_N_density_weighted_ecdf_loglog.png",
                    dpi=300,
                    bbox_inches="tight"
                )
                plt.close()
            else:
                print(
                    f"{parent_nuc}: No valid entries for absolute error density-weighted ECDF.",
                    file=knn_file
                )
            if np.any(del_N_by_N_v_N_log_log_mask):

                sort_frac_delta_N_indices = np.argsort(abs_fractional_errors[del_N_by_N_v_N_log_log_mask])
                sorted_frac_delta_N_errors = abs_fractional_errors[del_N_by_N_v_N_log_log_mask][sort_frac_delta_N_indices]
                sorted_frac_delta_N_densities = true_densities[del_N_by_N_v_N_log_log_mask][sort_frac_delta_N_indices]

                cumulative_frac_delta_N_density = np.cumsum(sorted_frac_delta_N_densities)
                cumulative_frac_delta_N_fraction = cumulative_frac_delta_N_density / np.sum(sorted_frac_delta_N_densities) #fraction calculated with only nonzero entries in denominator

                plt.figure()
                plt.step(sorted_frac_delta_N_errors, cumulative_frac_delta_N_fraction, where="post")
                plt.xscale("log")
                plt.yscale("log")
                plt.xlabel("Fractional Error |Predicted Density - True Density| / True Density")
                plt.ylabel("Cumulative Fraction of Total Number Density")
                plt.title("Number-Density-Weighted ECDF (Log-Log)")
                plt.grid(True, which="both")
                plt.tight_layout()
                plt.savefig(f"{plot_folder}/{parent_nuc}_frac_delta_N_density_weighted_ecdf_loglog.png", dpi=300, bbox_inches="tight")
                plt.show()
                plt.close()
            else:
                print(
                    f"{parent_nuc}: No valid entries for fractional error density-weighted ECDF.",
                    file=knn_file
                )

            # ---------------------------- number density-weighted ecdf ends
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(true_densities, delta_N, s=5, alpha=0.3)
            ax.set_xlabel("True Number Density (atoms/cm³)")
            ax.set_ylabel(r"Error $\Delta$N (atoms/cm³)")
            ax.set_title(f"KNN Prediction Error vs True Number Density: {parent_nuc}")
            fig.tight_layout()
            fig.savefig(f"{plot_folder}/{parent_nuc}_delta_N_vs_N.png", dpi=300)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(true_densities, abs_delta_N, s=5, alpha=0.3)
            ax.set_xlabel("True Number Density (atoms/cm³)")
            ax.set_ylabel(r"Absolute Error $\Delta$N (atoms/cm³)")
            ax.set_title(f"Absolute KNN Prediction Error vs True Number Density: {parent_nuc}")
            fig.tight_layout()
            fig.savefig(f"{plot_folder}/{parent_nuc}_abs_delta_N_vs_N.png", dpi=300)
            plt.close(fig)

    diff = list(set(parent_nucs) - set(available_parent_nucs))
    print(diff)


    fig, ax = plt.subplots(figsize=(24, 10), constrained_layout=True)
    ax.bar(range(len(available_parent_nucs)), parent_rmse)
    ax.set_xlabel("Parent nuclide")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE for Each Parent Nuclide")
    ax.set_xticks(range(len(available_parent_nucs)))
    ax.set_xticklabels(available_parent_nucs, rotation=90, fontsize=7)
    fig.savefig(f"{plot_folder}/pre_approx_parent_rmse.png", dpi=300)
    plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(24, 10), constrained_layout=True)
    ax.bar(range(len(available_parent_nucs)), parent_mae)
    ax.set_xlabel("Parent nuclide")
    ax.set_ylabel("MAE")
    ax.set_title("MAE for Each Parent Nuclide")
    ax.set_xticks(range(len(available_parent_nucs)))
    ax.set_xticklabels(available_parent_nucs, rotation=90, fontsize=7)
    fig.savefig(f"{plot_folder}/pre_approx_parent_mae.png", dpi=300)
    plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(24, 10), constrained_layout=True)
    ax.bar(range(len(available_parent_nucs)), parent_r2)
    ax.set_xlabel("Parent nuclide")
    ax.set_ylabel("R²")
    ax.set_title("R² for Each Parent Nuclide")
    ax.set_xticks(range(len(available_parent_nucs)))
    ax.set_xticklabels(available_parent_nucs, rotation=90, fontsize=7)
    fig.savefig(f"{plot_folder}/pre_approx_parent_r2.png", dpi=300)
    plt.show()   
    plt.close() 

if __name__ == "__main__":
    main()