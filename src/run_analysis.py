"""Run from any directory: python src/run_analysis.py --offline."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import json
from pathlib import Path

from nyc_airbnb.data import prepare_data
from nyc_airbnb.modeling import evaluate_models, estimate_associations, summarize
from nyc_airbnb.scope import primary_mask


def main():
    parser = argparse.ArgumentParser(description="Reproduce NYC Airbnb spatial pricing analysis")
    parser.add_argument("--offline", action="store_true", help="Use cached inputs only; fail if any are missing")
    parser.add_argument("--prepare-only", action="store_true", help="Validate data and build spatial features without models")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    tables = results / "tables"
    processed = root / "data/processed"
    tables.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    all_listings, tracts, stations, quality = prepare_data(root, offline=args.offline)
    listings = all_listings[primary_mask(all_listings)].copy().reset_index(drop=True)
    quality.update({"priced_spatial_listings": len(all_listings), "primary_listing_count": len(listings),
                    "hospitality_excluded": len(all_listings) - len(listings),
                    "primary_scope": "Exclude Hotel room or property_type containing hotel, hostel, resort; no price-tail trimming",
                    "primary_median_minimum_nights": float(listings.minimum_nights.median())})
    listings.to_csv(processed / "listings.csv", index=False)
    all_listings.to_csv(processed / "all_listings.csv", index=False)
    tracts.to_crs(4326).to_file(processed / "tracts.geojson", driver="GeoJSON")
    stations.to_csv(processed / "stations.csv", index=False)
    (results / "source_manifest.json").write_text(json.dumps(quality["source_manifest"], indent=2) + "\n", encoding="utf-8")
    (results / "data_quality.json").write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")
    if args.prepare_only:
        print(f"Prepared {len(listings):,} listings")
        return
    metrics, predictions = evaluate_models(listings, tables)
    predictions.to_csv(processed / "oof_predictions.csv.gz", index=False)
    associations = estimate_associations(listings, tables)
    summary = summarize(listings, tables, predictions)
    catalog_dir = tables / "full_catalog"
    catalog_dir.mkdir(exist_ok=True)
    catalog_metrics, catalog_predictions = evaluate_models(all_listings, catalog_dir)
    catalog_metrics.to_csv(tables / "full_catalog_model_metrics.csv", index=False)
    catalog_quality = {
        "listing_count": len(all_listings), "hotel_room_median_usd": float(all_listings.loc[all_listings.room_type.eq("Hotel room"), "price"].median()),
        "above_5000_count": int((all_listings.price > 5000).sum()),
        "above_5000_sum_price_share_pct": float(all_listings.loc[all_listings.price > 5000, "price"].sum() / all_listings.price.sum() * 100),
        "note": "Sum of asking prices is a data-quality diagnostic, not revenue. Suspicious high asks are not corrected. Full catalog is a separate sensitivity population.",
    }
    summary.update({"quality": quality, "models": metrics.to_dict(orient="records"),
                    "associations": associations.to_dict(orient="records"), "seed": 42,
                    "full_catalog": catalog_quality, "full_catalog_models": catalog_metrics.to_dict(orient="records")})
    (results / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    from nyc_airbnb.presentation import create_figures, create_map
    create_figures(root)
    create_map(root)
    print(f"Results saved to {results}")


if __name__ == "__main__":
    main()
