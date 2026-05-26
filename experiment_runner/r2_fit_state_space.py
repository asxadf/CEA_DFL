# experiment_runner/r2_fit_state_space.py

"""
Fit incremental-form dynamics matrices/vectors (M, N, O, m_vec) from generated one-step digital twin data.

Example:
python3 experiment_runner/r2_fit_state_space.py --season cold
python3 experiment_runner/r2_fit_state_space.py --season warm

python3 experiment_runner/r2_fit_state_space.py --season cold &
python3 experiment_runner/r2_fit_state_space.py --season warm &

wait
"""

# INCREMENTAL FORM: x_{k+1} = x_k + dX_k, dX_k = M x_k + N u_k + O d_k + m
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.builders.builder_digital_twin_one_step import load_parameters
from src.utils.vector_order import (
    get_control_indices,
    get_control_names_in_order,
    get_disturbance_names_in_order,
    get_num_controls,
    get_num_disturbances,
    get_num_states,
    get_state_names_in_order,
)

SEED = 123
PARAM_YAML_PATH = (Path(__file__).resolve().parents[1] / "configs/var_and_param_keeper.yaml").resolve()
OUT_BASE_DIR = (Path(__file__).resolve().parents[1] / "experiment_result/r2_fit_state_space").resolve()
TRAIN_FRAC = 0.8
ENET_ALPHA = 1e-3
ENET_L1_RATIO = 0.5
FIT_INTERCEPT = True
USE_ALPHA_GRID = True
ALPHA_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

max_iter = 1000000

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit incremental-form matrices and vectors.")
    parser.add_argument("--season", choices=("cold", "warm"), default="cold")
    return parser.parse_args()


def _print_matrix(title: str, mat: np.ndarray, row_labels: list[str], col_labels: list[str]) -> None:
    row_w = 8
    col_w = max(12, max(len(c) for c in col_labels) + 2)
    print(f"\n{title}")
    print(f"{'':<{row_w}}" + " ".join(f"{c:>{col_w}}" for c in col_labels))
    for rlab, row in zip(row_labels, mat):
        print(f"{rlab:<{row_w}}" + " ".join(f"{v:>{col_w}.4f}" for v in row))


def _print_vector(title: str, labels: list[str], vals: np.ndarray) -> None:
    w = 10
    print(f"\n{title}")
    print(f"{'':<8}" + "".join(f"{c:>{w}}" for c in labels))
    print(f"{'val':<8}" + "".join(f"{v:>{w}.4f}" for v in vals))


def _print_column_vector(title: str, labels: list[str], vals: np.ndarray) -> None:
    label_w = max(8, max(len(c) for c in labels) + 2)
    val_w = 12
    print(f"\n{title}")
    print(f"{'row':<{label_w}}{'value':>{val_w}}")
    for lbl, val in zip(labels, vals):
        print(f"{lbl:<{label_w}}{val:>{val_w}.4f}")


def _require_columns(df: pd.DataFrame, cols: list[str], block_name: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required {block_name} columns in data.csv: {missing}")


def main() -> None:
    args = parse_args()
    season = args.season
    DATA_CSV_PATH = (
        Path(__file__).resolve().parents[1]
        / "experiment_result/r1_generate_data_from_twin"
        / season
        / "data.csv"
    ).resolve()

    if not DATA_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Missing generated data file: {DATA_CSV_PATH}. "
            f"Run r1_generate_data_from_twin.py for season '{season}' first."
        )

    param_keeper = load_parameters(PARAM_YAML_PATH)
    x_names = get_state_names_in_order(param_keeper)
    u_names = get_control_names_in_order(param_keeper)
    d_names = get_disturbance_names_in_order(param_keeper)
    num_x = get_num_states(param_keeper)
    num_u = get_num_controls(param_keeper)
    num_d = get_num_disturbances(param_keeper)
    row_labels = [f"d{name}" for name in x_names]
    m_vec_cols = [f"m_{name}" for name in x_names]
    x_ini_csv_cols = [f"X_ini_{name}" for name in x_names]
    u_csv_cols = u_names
    d_csv_cols = [f"D_{name}" for name in d_names]
    dx_csv_cols = [f"DX_d{name}" for name in x_names]

    rng = np.random.default_rng(SEED)
    data_df = pd.read_csv(DATA_CSV_PATH)
    _require_columns(data_df, x_ini_csv_cols, "X_ini")
    _require_columns(data_df, u_csv_cols, "U")
    _require_columns(data_df, d_csv_cols, "D")
    _require_columns(data_df, dx_csv_cols, "DX")

    X_INI = data_df[x_ini_csv_cols].to_numpy(dtype=float)
    U = data_df[u_csv_cols].to_numpy(dtype=float)
    D = data_df[d_csv_cols].to_numpy(dtype=float)
    DX = data_df[dx_csv_cols].to_numpy(dtype=float)

    n_samples = X_INI.shape[0]
    Phi = np.concatenate([X_INI, U, D], axis=1)

    Y = DX

    idx = rng.permutation(n_samples)
    n_train = int(TRAIN_FRAC * n_samples)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    Phi_train, Phi_test = Phi[train_idx], Phi[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    X_INI_test = X_INI[test_idx]

    n_outputs = Y.shape[1]
    chosen_alpha_list = [float(ENET_ALPHA) for _ in range(n_outputs)]
    models = []
    for j in range(n_outputs):
        y_train_j = Y_train[:, j]
        y_test_j = Y_test[:, j]

        chosen_alpha_j = float(ENET_ALPHA)
        if USE_ALPHA_GRID:
            best_rmse_j = np.inf
            for a in ALPHA_GRID:
                m_try = ElasticNet(
                    alpha=float(a),
                    l1_ratio=ENET_L1_RATIO,
                    fit_intercept=FIT_INTERCEPT,
                    max_iter=max_iter,
                    tol=1e-6,
                )
                m_try.fit(Phi_train, y_train_j)
                y_test_pred_try = m_try.predict(Phi_test)
                rmse_try = float(np.sqrt(np.mean((y_test_pred_try - y_test_j) ** 2)))
                if rmse_try < best_rmse_j:
                    best_rmse_j = rmse_try
                    chosen_alpha_j = float(a)


        model_j = ElasticNet(
            alpha=chosen_alpha_j,
            l1_ratio=ENET_L1_RATIO,
            fit_intercept=FIT_INTERCEPT,
            max_iter=max_iter,
            tol=1e-6,
        )
        model_j.fit(Phi_train, y_train_j)
        models.append(model_j)
        chosen_alpha_list[j] = chosen_alpha_j

    W_raw = np.asarray([m.coef_ for m in models], dtype=float)
    b_raw = np.asarray([m.intercept_ for m in models], dtype=float) if FIT_INTERCEPT else np.zeros(num_x, dtype=float)

    M = W_raw[:, 0:num_x]
    N_mat = W_raw[:, num_x:num_x + num_u]
    O = W_raw[:, num_x + num_u:num_x + num_u + num_d]
    m_vec = b_raw

    A = np.eye(num_x) + M
    eig_A = np.linalg.eigvals(A)
    max_abs_eig_A = float(np.max(np.abs(eig_A)))

    DX_train_pred = np.column_stack([m.predict(Phi_train) for m in models])
    DX_test_pred = np.column_stack([m.predict(Phi_test) for m in models])

    train_rmse_dx = np.sqrt(np.mean((DX_train_pred - Y_train) ** 2, axis=0))
    test_rmse_dx = np.sqrt(np.mean((DX_test_pred - Y_test) ** 2, axis=0))

    X1_true = X_INI_test + Y_test
    X1_pred = X_INI_test + DX_test_pred
    test_rmse_x1 = np.sqrt(np.mean((X1_pred - X1_true) ** 2, axis=0))

    u_idx = get_control_indices(param_keeper)
    corr_u_fan_u_nat = float(np.corrcoef(U[:, u_idx["U_fan"]], U[:, u_idx["U_nat"]])[0, 1])

    out_dir = OUT_BASE_DIR / season
    out_dir.mkdir(parents=True, exist_ok=True)

    for item in out_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    pd.DataFrame(M, index=row_labels, columns=x_names).to_csv(out_dir / "M.csv", index_label="row", float_format="%.4f")
    pd.DataFrame(N_mat, index=row_labels, columns=u_names).to_csv(out_dir / "N.csv", index_label="row", float_format="%.4f")
    pd.DataFrame(O, index=row_labels, columns=d_names).to_csv(out_dir / "O.csv", index_label="row", float_format="%.4f")
    pd.DataFrame(A, index=x_names, columns=x_names).to_csv(out_dir / "A.csv", index_label="row", float_format="%.4f")
    pd.DataFrame({"value": m_vec}, index=m_vec_cols).to_csv(out_dir / "m_vec.csv", index_label="row", float_format="%.4f")
    pd.DataFrame({"output": row_labels, "chosen_alpha": chosen_alpha_list}).to_csv(
        out_dir / "chosen_alpha.csv",
        index=False,
    )

    meta = {
        "data_path": str(DATA_CSV_PATH),
        "N": int(n_samples),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "enet_alpha": float(ENET_ALPHA),
        "enet_l1_ratio": float(ENET_L1_RATIO),
        "chosen_alpha_list": [float(a) for a in chosen_alpha_list],
        "train_rmse_dx": train_rmse_dx.tolist(),
        "test_rmse_dx": test_rmse_dx.tolist(),
        "test_rmse_x1": test_rmse_x1.tolist(),
        "max_abs_eig_A": max_abs_eig_A,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n========== INCREMENTAL FORM: x_{k+1} = x_k + dX_k, dX_k = M x_k + N u_k + O d_k + m_vec | ElasticNet (per-output) ==========")
    print(f"{'data_path':<20}{DATA_CSV_PATH}")
    print(f"{'N':<20}{n_samples:>10d}")
    print(f"{'train_size':<20}{len(train_idx):>10d}")
    print(f"{'test_size':<20}{len(test_idx):>10d}")
    print(f"{'enet_alpha':<20}{ENET_ALPHA:>10.4g}")
    for j, lbl in enumerate(row_labels):
        print(f"{f'chosen_alpha_{lbl}':<20}{chosen_alpha_list[j]:>10.4g}")
    print(f"{'enet_l1_ratio':<20}{ENET_L1_RATIO:>10.4g}")
    print(f"{'fit_intercept':<20}{str(FIT_INTERCEPT):>10}")
    print(f"{'M shape':<20}{str(M.shape):>10}")
    print(f"{'N shape':<20}{str(N_mat.shape):>10}")
    print(f"{'O shape':<20}{str(O.shape):>10}")
    print(f"{'m_vec shape':<20}{str(m_vec.shape):>10}")

    _print_matrix("M", M, row_labels, x_names)
    _print_matrix("N", N_mat, row_labels, u_names)
    _print_matrix("O", O, row_labels, d_names)
    _print_column_vector("m_vec", m_vec_cols, m_vec)

    print(f"\n{'max(|eig(A)|)':<20}{max_abs_eig_A:>10.4f}")
    _print_column_vector("Train RMSE DX", row_labels, train_rmse_dx)
    _print_column_vector("Test RMSE DX", row_labels, test_rmse_dx)
    _print_column_vector("Test RMSE X1", x_names, test_rmse_x1)
    print(f"\n{'corr(U_fan,U_nat)':<20}{corr_u_fan_u_nat:>10.4f}")


if __name__ == "__main__":
    main()
