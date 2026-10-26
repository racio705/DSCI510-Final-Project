"""Fixed model comparisons, out-of-fold evaluation and observational inference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
BASE_NUMERIC = ["accommodates", "bedrooms", "beds", "bathrooms", "log_minimum_nights",
                "log_review_count", "review_scores_rating", "log_host_listings_count"]
CONTEXT_NUMERIC = ["latitude", "longitude", "log_income", "population_density_km2"]
TRANSIT_NUMERIC = ["dist_km", "stations_within_800m"]
CATEGORICAL = ["room_type", "borough", "property_type"]
MODEL_NAMES = ["Global median", "Room + borough median", "Distance + income",
               "Listing controls (ridge)", "Full without transit", "Full with transit"]


def feature_pipeline(numeric: list[str], categorical: list[str], *, nonlinear=False) -> Pipeline:
    """Imputers and encoders are fitted on each training fold only."""
    transforms = [("numeric", Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ]), numeric)]
    if categorical:
        transforms.append(("category", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    estimator = HistGradientBoostingRegressor(
        max_iter=140, learning_rate=0.07, max_leaf_nodes=15, min_samples_leaf=35,
        l2_regularization=1.0, early_stopping=False, random_state=SEED,
    ) if nonlinear else Ridge(alpha=20.0)
    return Pipeline([("features", ColumnTransformer(transforms)), ("model", estimator)])


def scores(actual: np.ndarray, predicted_log: np.ndarray) -> dict:
    """Dollar error on the original price scale; R² on log1p(price)."""
    prediction = np.maximum(np.expm1(predicted_log), 0)
    return {
        "n": len(actual),
        "mae_usd": float(mean_absolute_error(actual, prediction)),
        "rmse_usd": float(np.sqrt(mean_squared_error(actual, prediction))),
        "median_ae_usd": float(np.median(np.abs(actual - prediction))),
        "r2_log": float(r2_score(np.log1p(actual), predicted_log)),
    }


def evaluate_models(frame: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same five folds for all six models, repeated under two validation designs.

    Spatial blocks are 2 km UTM grid cells. Neighboring cells can touch, so this
    is a local geographic transfer test, not buffered or temporal validation.
    """
    all_numeric = BASE_NUMERIC + CONTEXT_NUMERIC + TRANSIT_NUMERIC
    frame = frame.copy().reset_index(drop=True)
    for column in all_numeric:
        if column not in frame:
            frame[column] = np.nan
    for column in CATEGORICAL:
        frame[column] = frame[column].fillna("Unknown").astype(object)
    x, y = frame, frame["log_price"].to_numpy()
    actual = frame["price"].to_numpy()
    folds, metrics, predictions = [], [], []
    schemes = {
        "Random 5-fold": list(KFold(5, shuffle=True, random_state=SEED).split(x)),
        "Spatial 2 km 5-fold": list(GroupKFold(5).split(x, y, groups=x["spatial_block"])),
    }
    for validation, splits in schemes.items():
        oof = {name: np.full(len(x), np.nan) for name in MODEL_NAMES}
        fold_ids = np.full(len(x), -1)
        for fold, (train, test) in enumerate(splits, start=1):
            if validation.startswith("Spatial"):
                assert set(x.iloc[train]["spatial_block"]).isdisjoint(x.iloc[test]["spatial_block"])
            fold_ids[test] = fold
            train_x, test_x, train_y = x.iloc[train], x.iloc[test], y[train]
            for name in MODEL_NAMES:
                print(f"{validation}, fold {fold}/5, {name}", flush=True)
                if name == "Global median":
                    predicted = np.full(len(test), np.median(train_y))
                elif name == "Room + borough median":
                    medians = train_x.assign(target=train_y).groupby(
                        ["room_type", "borough"], observed=True)["target"].median()
                    keys = pd.MultiIndex.from_frame(test_x[["room_type", "borough"]])
                    predicted = medians.reindex(keys).fillna(np.median(train_y)).to_numpy()
                else:
                    if name == "Distance + income":
                        model = feature_pipeline(["dist_km", "income_10k"], [])
                    elif name == "Listing controls (ridge)":
                        model = feature_pipeline(BASE_NUMERIC, CATEGORICAL)
                    else:
                        numeric = BASE_NUMERIC + CONTEXT_NUMERIC
                        if name == "Full with transit":
                            numeric += TRANSIT_NUMERIC
                        model = feature_pipeline(numeric, CATEGORICAL, nonlinear=True)
                    model.fit(train_x, train_y)
                    predicted = model.predict(test_x)
                oof[name][test] = predicted
                folds.append({"validation": validation, "fold": fold, "model": name,
                              "n_train": len(train), "n_test": len(test),
                              **scores(actual[test], predicted)})
        for name, predicted in oof.items():
            assert np.isfinite(predicted).all() and (fold_ids > 0).all()
            metrics.append({"validation": validation, "model": name, **scores(actual, predicted)})
        predictions.append(pd.DataFrame({
            "id": x["id"], "GEOID": x["GEOID"], "validation": validation,
            "fold": fold_ids, "price": actual, "log_price": y,
            "predicted_log_full": oof["Full with transit"],
            "predicted_log_no_transit": oof["Full without transit"],
        }))
    result = pd.DataFrame(metrics)
    result.to_csv(output / "model_metrics.csv", index=False)
    pd.DataFrame(folds).to_csv(output / "fold_metrics.csv", index=False)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    return result, prediction_frame


def estimate_associations(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Linear distance association with tract-clustered 95% confidence intervals.

    This is descriptive inference. A fixed slope and clustered standard errors
    do not remove unobserved confounding or establish transit's causal effect.
    """
    d = frame.copy()
    d["ln_price"] = np.log(d["price"])
    for col in ["room_type", "borough", "neighbourhood_cleansed"]:
        d[col] = d[col].fillna("Unknown").astype(object)
    d["neighborhood_fe"] = (d["borough"].astype(str) + "|" + d["neighbourhood_cleansed"].astype(str)).astype(object)
    controls = ("C(room_type) + C(borough) + accommodates + bedrooms + bathrooms + "
                "log_minimum_nights + log_review_count + review_scores_rating")
    neighborhood_controls = controls.replace("C(borough) + ", "")
    # Compare the first three specifications on the same complete-case sample.
    # Neighborhood effects already absorb borough effects; including both would
    # make the design matrix rank deficient.
    d = d.dropna(subset=["dist_km", "log_income", "ln_price", "GEOID", "accommodates", "bedrooms",
                         "bathrooms", "log_minimum_nights", "log_review_count", "review_scores_rating"])
    specifications = [
        ("Distance + income", "ln_price ~ dist_km + log_income", d),
        ("Room, size + borough controls", f"ln_price ~ dist_km + log_income + {controls}", d),
        ("Neighborhood fixed effects", f"ln_price ~ dist_km + log_income + {neighborhood_controls} + C(neighborhood_fe)", d),
        ("Within 3 km of subway", f"ln_price ~ dist_km + log_income + {neighborhood_controls} + C(neighborhood_fe)", d[d.dist_km <= 3]),
        ("Central 98% price sensitivity", f"ln_price ~ dist_km + log_income + {neighborhood_controls} + C(neighborhood_fe)",
         d[d.price.between(d.price.quantile(.01), d.price.quantile(.99))]),
    ]
    rows = []
    for name, formula, selected in specifications:
        required = ["dist_km", "log_income", "ln_price", "GEOID"]
        if name != "Distance + income":
            required += ["accommodates", "bedrooms", "bathrooms", "log_minimum_nights",
                         "log_review_count", "review_scores_rating"]
        complete = selected.dropna(subset=required).copy()
        fit = smf.ols(formula, data=complete).fit(
            cov_type="cluster", cov_kwds={"groups": complete["GEOID"], "use_correction": True}, use_t=True)
        beta = float(fit.params["dist_km"])
        lower, upper = fit.conf_int().loc["dist_km"]
        rows.append({"specification": name, "n": int(fit.nobs),
                     "tract_clusters": complete["GEOID"].nunique(),
                     "coefficient_per_km": beta, "percent_per_500m": float(100 * np.expm1(beta * .5)),
                     "ci_lower_pct": float(100 * np.expm1(lower * .5)),
                     "ci_upper_pct": float(100 * np.expm1(upper * .5)),
                     "p_value": float(fit.pvalues["dist_km"]), "r2_in_sample_log": float(fit.rsquared)})
    result = pd.DataFrame(rows)
    result.to_csv(output / "coefficient_estimates.csv", index=False)
    return result


def moran_permutation(values: np.ndarray, coords: np.ndarray, *, permutations=499) -> dict:
    """Row-standardized symmetric 8-nearest-neighbor Moran I, seeded permutations."""
    n = len(values)
    k = min(8, n - 1)
    neighbors = cKDTree(coords).query(coords, k=k + 1)[1][:, 1:]
    weights = csr_matrix((np.ones(n * k), (np.repeat(np.arange(n), k), neighbors.ravel())), shape=(n, n))
    weights = weights.maximum(weights.T)
    weights = weights.multiply(1 / np.asarray(weights.sum(axis=1)).ravel()[:, None]).tocsr()
    z = np.asarray(values) - np.mean(values)
    denominator = z @ z
    observed = float((z @ (weights @ z)) / denominator)
    rng = np.random.default_rng(SEED)
    permuted = np.empty(permutations)
    for i in range(permutations):
        shuffled = rng.permutation(z)
        permuted[i] = shuffled @ (weights @ shuffled) / denominator
    expected = -1 / (n - 1)
    p = (1 + np.count_nonzero(np.abs(permuted - expected) >= abs(observed - expected))) / (permutations + 1)
    return {"moran_i": observed, "p_two_sided": float(p), "tracts": n,
            "permutations": permutations, "weights": "symmetrized 8-nearest-tract-centroids, row-standardized"}


def summarize(frame: pd.DataFrame, output: Path, predictions: pd.DataFrame) -> dict:
    """Describe the selected sample, retain group counts, and test coordinate-noise sensitivity."""
    borough = frame.groupby("borough", observed=True).agg(
        n=("price", "size"), median_price=("price", "median"), median_distance_m=("dist_km", lambda x: x.median() * 1000),
        within_500m_pct=("dist_km", lambda x: (x <= .5).mean() * 100))
    borough.reset_index().to_csv(output / "borough_summary.csv", index=False)
    d = frame.copy()
    d["distance_band"] = pd.cut(d["dist_km"], bins=[0, .25, .5, 1, 2, np.inf],
                                labels=["0–250 m", "250–500 m", "500 m–1 km", "1–2 km", ">2 km"], include_lowest=True)
    d.groupby(["room_type", "distance_band"], observed=True).agg(n=("price", "size"), median_price=("price", "median")).reset_index().to_csv(
        output / "distance_segments.csv", index=False)
    frame.groupby(["borough", "room_type"], observed=True).agg(n=("price", "size"), median_price=("price", "median")).reset_index().to_csv(
        output / "room_borough_segments.csv", index=False)
    threshold = pd.DataFrame({"threshold_m": [350, 500, 650, 800, 1000],
                              "share_pct": [(frame.dist_km <= t / 1000).mean() * 100 for t in [350, 500, 650, 800, 1000]]})
    threshold.to_csv(output / "access_threshold_sensitivity.csv", index=False)
    spatial = predictions[predictions.validation == "Spatial 2 km 5-fold"].copy()
    spatial["residual"] = spatial.log_price - spatial.predicted_log_full
    joined = frame[["id", "x_m", "y_m"]].merge(spatial, on="id", validate="one_to_one")
    tract = joined.groupby("GEOID").agg(n=("price", "size"), x=("x_m", "mean"), y=("y_m", "mean"),
                                        log_price=("log_price", "mean"), residual=("residual", "mean"))
    tract = tract[tract.n >= 5]
    diagnostic = {
        "spatial_autocorrelation": {
            "observed_log_price": moran_permutation(tract.log_price.to_numpy(), tract[["x", "y"]].to_numpy()),
            "oof_residual": moran_permutation(tract.residual.to_numpy(), tract[["x", "y"]].to_numpy()),
            "note": "Tracts with >=5 selected listings; coordinates are listing means. Residuals are full-with-transit spatial-fold OOF. Exploratory diagnostic, not model selection.",
        },
        "listing_count": len(frame), "median_price_usd": float(frame.price.median()),
        "median_distance_m": float(frame.dist_km.median() * 1000),
        "within_500m_pct": float((frame.dist_km <= .5).mean() * 100),
        "within_350m_pct": float((frame.dist_km <= .35).mean() * 100),
        "within_650m_pct": float((frame.dist_km <= .65).mean() * 100),
        "spatial_blocks": frame.spatial_block.nunique(),
        "price_p99_usd": float(frame.price.quantile(.99)), "price_max_usd": float(frame.price.max()),
    }
    return diagnostic
