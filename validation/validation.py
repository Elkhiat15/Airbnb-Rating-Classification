import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPORT_DIR = Path("validation_report")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(REPORT_DIR / "validation.log"),
    ],
)
logger = logging.getLogger(__name__)


class DataValidator:
    VALID_ROOM_TYPES = {
        "Entire home/apt", "Private room", "Shared room"
    }
    VALID_CITIES = {
        "New York", 
        "Los Angeles", 
        "San Francisco", 
        "Washington DC", 
        "Chicago", 
        "Boston"
    }
    VALID_PROPERTY_GROUPS = {
        "Apartment", "House", "Condo", "Townhouse", "Loft", 
        "Guesthouse", "Hotel", "Shared Room", "Unique Stay"
    }
    
    BOOLEAN_VALUES = {True, False}

    PRICE_MIN, PRICE_MAX            = 10.0, 10_000.0
    RATING_MIN, RATING_MAX          = 0.0, 5.0
    ACCOMMODATES_MIN, ACCOMMODATES_MAX = 1, 16
    BATHROOMS_MIN, BATHROOMS_MAX    = 0.0, 10.0
    BEDROOMS_MIN, BEDROOMS_MAX      = 0.0, 12.0
    BEDS_MIN, BEDS_MAX              = 0.0, 20.0
    REVIEWS_MIN, REVIEWS_MAX        = 0, 10_000
    RESPONSE_MIN, RESPONSE_MAX      = 0.0, 100.0
    LAT_MIN, LAT_MAX                = 24.0, 50.0    
    LON_MIN, LON_MAX                = -125.0, -66.0

    REQUIRED_COLUMNS = [
        "property_type", "room_type", "amenities", "accommodates",
        "bathrooms", "city", "host_identity_verified", "host_response_rate",
        "latitude", "longitude", "neighbourhood", "number_of_reviews",
        "bedrooms", "beds", "price", "review_scores_rating",
    ]

    NUMERIC_COLS = [
        "price", "accommodates", "bathrooms", "bedrooms", "beds",
        "number_of_reviews", "host_response_rate",
        "review_scores_rating", "latitude", "longitude",
    ]

    CATEGORICAL_COLS = [
        "property_type", "room_type", "city",
        "host_identity_verified", "neighbourhood",
    ]

    IQR_MULTIPLIER   = 1.5
    ZSCORE_THRESHOLD = 3.0


    def __init__(self, df: pd.DataFrame, dataset_path: str = ""):
        self.df = df.copy()
        self.dataset_path = dataset_path
        self.n_rows, self.n_cols = self.df.shape
        self.results: dict = {}
        self._coerce_types()
        logger.info(
            f"DataValidator initialised — {self.n_rows:,} rows x {self.n_cols} columns"
        )

    def _coerce_types(self) -> None:
        for col in self.NUMERIC_COLS:
            if col == 'host_response_rate': 
                self.df[col] = self.df[col].str.replace('%', '').astype(float)
                continue
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

    @staticmethod
    def _iqr_bounds(series: pd.Series, multiplier: float = 1.5):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        return q1 - multiplier * iqr, q3 + multiplier * iqr

    @staticmethod
    def _pct(count: int, total: int) -> float:
        return round(count / total * 100, 3) if total else 0.0

    def _col(self, name: str) -> pd.Series:
        return self.df[name]

    def _pass_fail(self, condition: bool) -> str:
        return "PASS" if condition else "FAIL"

    def check_accuracy(self) -> dict:
        logger.info("Running [1] Accuracy checks...")
        report = {"dimension": "Accuracy", "checks": []}
        checks = report["checks"]

        def range_check(col, lo, hi, mostly=1.0):
            if col not in self.df.columns:
                return
            s = self._col(col).dropna()
            violations = int(((s < lo) | (s > hi)).sum())
            total = len(s)
            violation_pct = self._pct(violations, total)
            threshold_pct = (1 - mostly) * 100
            passed = violation_pct <= threshold_pct
            checks.append({
                "check": f"{col} in [{lo}, {hi}]",
                "column": col,
                "type": "range",
                "valid_min": lo,
                "valid_max": hi,
                "violations": violations,
                "violation_pct": violation_pct,
                "threshold_pct": threshold_pct,
                "status": self._pass_fail(passed),
            })

        def set_check(col, valid_set, mostly=1.0):
            if col not in self.df.columns:
                return
            s = self._col(col).dropna()
            normalised = s.astype(str).str.strip()
            valid_lower = {str(v).strip() for v in valid_set}
            violations = int((~normalised.isin(valid_lower)).sum())
            total = len(s)
            violation_pct = self._pct(violations, total)
            threshold_pct = (1 - mostly) * 100
            passed = violation_pct <= threshold_pct
            invalid_vals = list(
                normalised[~normalised.isin(valid_lower)].unique()[:10]
            )
            checks.append({
                "check": f"{col} in valid set",
                "column": col,
                "type": "set_membership",
                "valid_set": list(valid_set),
                "violations": violations,
                "violation_pct": violation_pct,
                "threshold_pct": threshold_pct,
                "sample_invalid_values": invalid_vals,
                "status": self._pass_fail(passed),
            })

        range_check("price", self.PRICE_MIN, self.PRICE_MAX)
        range_check("review_scores_rating", self.RATING_MIN, self.RATING_MAX)
        range_check("accommodates", self.ACCOMMODATES_MIN, self.ACCOMMODATES_MAX)
        range_check("bathrooms", self.BATHROOMS_MIN, self.BATHROOMS_MAX)
        range_check("bedrooms", self.BEDROOMS_MIN, self.BEDROOMS_MAX)
        range_check("beds", self.BEDS_MIN, self.BEDS_MAX)
        range_check("number_of_reviews", self.REVIEWS_MIN, self.REVIEWS_MAX)
        range_check("host_response_rate", self.RESPONSE_MIN, self.RESPONSE_MAX)
        range_check("latitude", self.LAT_MIN, self.LAT_MAX)
        range_check("longitude", self.LON_MIN, self.LON_MAX)

        set_check("room_type", self.VALID_ROOM_TYPES)
        set_check("city", self.VALID_CITIES)
        set_check("host_identity_verified", self.BOOLEAN_VALUES)
        set_check("property_type", self.VALID_PROPERTY_GROUPS)

        passed = sum(1 for c in checks if c["status"] == "PASS")
        report.update({
            "total_checks": len(checks),
            "passed": passed,
            "failed": len(checks) - passed,
            "overall_status": self._pass_fail(passed == len(checks)),
        })

        self.results["1_accuracy"] = report
        logger.info(f"  Accuracy: {passed}/{len(checks)} checks passed")
        return report

    def check_completeness(self) -> dict:
        logger.info("Running [2] Completeness checks...")

        # max allowed null %
        thresholds = {
            "price": 0.0,
            "room_type": 0.0,
            "city": 0.0,
            "review_scores_rating": 0.0,
            "latitude": 0.0,
            "longitude": 0.0,
            "accommodates": 0.0,
            "property_type": 0.0,
            "bedrooms": 0.0,
            "beds": 0.0,
            "bathrooms": 0.0,
            "host_identity_verified": 0.0,
            "neighbourhood": 0.0,
            "host_response_rate": 0.0,
            "amenities": 0.0,
            "number_of_reviews": 0.0,
        }

        checks = []
        for col in self.REQUIRED_COLUMNS:
            if col not in self.df.columns:
                checks.append({
                    "column": col,
                    "status": "FAIL",
                    "note": "Column missing from dataset",
                })
                continue
            null_count = int(self.df[col].isna().sum())
            null_pct = self._pct(null_count, self.n_rows)
            max_null_pct = thresholds.get(col, 20.0)
            passed = null_pct <= max_null_pct
            checks.append({
                "column": col,
                "null_count": null_count,
                "null_pct": null_pct,
                "max_allowed_null_pct": max_null_pct,
                "criticality": (
                    "critical" if max_null_pct == 0.0 else
                    "high" if max_null_pct <= 5.0 else "medium"
                ),
                "status": self._pass_fail(passed),
            })

        missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in self.df.columns]
        extra_cols   = [c for c in self.df.columns if c not in self.REQUIRED_COLUMNS]
        row_check_passed = self.n_rows >= 70_000
        schema_passed    = len(missing_cols) == 0
        passed_count = sum(1 for c in checks if c["status"] == "PASS")

        report = {
            "dimension": "Completeness",
            "row_count": {
                "actual": self.n_rows,
                "minimum_expected": 70_000,
                "status": self._pass_fail(row_check_passed),
            },
            "schema": {
                "expected_columns": len(self.REQUIRED_COLUMNS),
                "actual_columns": self.n_cols,
                "missing_columns": missing_cols,
                "extra_columns": extra_cols,
                "status": self._pass_fail(schema_passed),
            },
            "column_checks": checks,
            "total_checks": len(checks),
            "passed": passed_count,
            "failed": len(checks) - passed_count,
            "overall_missing_cells": int(self.df.isna().sum().sum()),
            "overall_missing_pct": round(
                self.df.isna().sum().sum() / (self.n_rows * self.n_cols) * 100, 3
            ),
            "overall_status": self._pass_fail(
                passed_count == len(checks) and schema_passed and row_check_passed
            ),
        }

        self.results["2_completeness"] = report
        logger.info(f"  Completeness: {passed_count}/{len(checks)} column checks passed")
        return report


    def check_consistency(self) -> dict:
        logger.info("Running [3] Consistency checks...")
        checks = []

        def add(name, violations, total, note="", mostly=1.0):
            viol_pct = self._pct(violations, total)
            threshold = (1 - mostly) * 100
            passed = viol_pct <= threshold
            checks.append({
                "check": name,
                "violations": violations,
                "violation_pct": viol_pct,
                "threshold_pct": threshold,
                "note": note,
                "status": self._pass_fail(passed),
            })

        n = self.n_rows

        
        if {"beds", "bedrooms"} <= set(self.df.columns):
            mask = self.df["beds"].notna() & self.df["bedrooms"].notna()
            sub = self.df[mask]
            add(
                "beds >= bedrooms",
                int((sub["beds"] < sub["bedrooms"]).sum()),
                len(sub),
                note="A listing should have at least as many beds as bedrooms",
                mostly=0.98,
            )

        if {"accommodates", "beds"} <= set(self.df.columns):
            mask = self.df["accommodates"].notna() & self.df["beds"].notna()
            sub = self.df[mask]
            add(
                "accommodates >= beds",
                int((sub["accommodates"] < sub["beds"]).sum()),
                len(sub),
                note="Guest capacity should not be less than number of beds",
                mostly=0.95,
            )

        if "number_of_reviews" in self.df.columns:
            add(
                "number_of_reviews >= 0",
                int((self.df["number_of_reviews"].dropna() < 0).sum()),
                n,
                note="Review count cannot be negative",
            )

        if "price" in self.df.columns:
            add(
                "price > 0",
                int((self.df["price"].dropna() <= 0).sum()),
                n,
                note="Price must be strictly positive",
            )

        if "host_response_rate" in self.df.columns:
            add(
                "host_response_rate in [0, 100]",
                int(
                    (
                        (self.df["host_response_rate"].dropna() < 0)
                        | (self.df["host_response_rate"].dropna() > 100)
                    ).sum()
                ),
                n,
                note="Response rate must be a valid percentage",
            )

        # Dtype consistency
        dtype_checks = {
            "price": "numeric", "accommodates": "numeric",
            "bathrooms": "numeric", "bedrooms": "numeric",
            "beds": "numeric", "number_of_reviews": "numeric",
            "latitude": "numeric", "longitude": "numeric",
        }
        for col, expected_kind in dtype_checks.items():
            if col in self.df.columns:
                actual = str(self.df[col].dtype)
                is_numeric = "float" in actual or "int" in actual
                checks.append({
                    "check": f"{col} is {expected_kind} dtype",
                    "column": col,
                    "actual_dtype": actual,
                    "note": "Non-numeric values coerced to NaN on load",
                    "status": self._pass_fail(is_numeric),
                })

        if {"latitude", "longitude"} <= set(self.df.columns):
            us_mask = (
                self.df["latitude"].between(self.LAT_MIN, self.LAT_MAX)
                & self.df["longitude"].between(self.LON_MIN, self.LON_MAX)
            )
            add(
                "coordinates within US bounding box",
                int((~us_mask).sum()),
                n,
                note="All listed cities are US cities — lat/lon must fall in US bounds",
                mostly=0.99,
            )

        passed = sum(1 for c in checks if c["status"] == "PASS")
        report = {
            "dimension": "Consistency",
            "checks": checks,
            "total_checks": len(checks),
            "passed": passed,
            "failed": len(checks) - passed,
            "overall_status": self._pass_fail(passed == len(checks)),
        }

        self.results["3_consistency"] = report
        logger.info(f"  Consistency: {passed}/{len(checks)} checks passed")
        return report


    def check_uniqueness(self) -> dict:
        logger.info("Running [4] Uniqueness checks...")
        checks = []
        n = self.n_rows

        exact_dup = int(self.df.duplicated().sum())
        checks.append({
            "check": "No exact full-row duplicates",
            "duplicate_count": exact_dup,
            "duplicate_pct": self._pct(exact_dup, n),
            "action_if_failed": "df.drop_duplicates()",
            "status": self._pass_fail(exact_dup == 0),
        })

        if {"latitude", "longitude"} <= set(self.df.columns):
            coord_dup = int(
                self.df.duplicated(subset=["latitude", "longitude"]).sum()
            )
            checks.append({
                "check": "Unique (latitude, longitude) pairs",
                "duplicate_count": coord_dup,
                "duplicate_pct": self._pct(coord_dup, n),
                "note": "Same coordinates may indicate duplicate listing or same building",
                "status": self._pass_fail(self._pct(coord_dup, n) <= 1.0),
            })

        fp_cols = ["latitude", "longitude", "price", "room_type"]
        if all(c in self.df.columns for c in fp_cols):
            fp_dup = int(self.df.duplicated(subset=fp_cols).sum())
            checks.append({
                "check": "Unique listing fingerprint (lat+lon+price+room_type)",
                "columns_used": fp_cols,
                "duplicate_count": fp_dup,
                "duplicate_pct": self._pct(fp_dup, n),
                "note": "Strong signal of duplicate listings from multiple sources",
                "status": self._pass_fail(self._pct(fp_dup, n) <= 0.5),
            })

        passed = sum(1 for c in checks if c["status"] == "PASS")
        report = {
            "dimension": "Uniqueness",
            "checks": checks,
            "total_checks": len(checks),
            "passed": passed,
            "failed": len(checks) - passed,
            "overall_status": self._pass_fail(passed == len(checks)),
        }

        self.results["4_uniqueness"] = report
        logger.info(f"  Uniqueness: {passed}/{len(checks)} checks passed")
        return report


    def check_outliers(self) -> dict:
        logger.info("Running [5] Outlier Detection...")
        col_results = {}
        OUTLIER_THRESHOLD_PCT = 5.0

        outlier_cols = [
            "price", "accommodates", "bathrooms", "bedrooms",
            "beds", "number_of_reviews", "host_response_rate",
            "review_scores_rating",
        ]

        for col in outlier_cols:
            if col not in self.df.columns:
                continue
            s = self.df[col].dropna()
            if len(s) == 0:
                continue

            lb, ub = self._iqr_bounds(s, self.IQR_MULTIPLIER)
            iqr_mask = (s < lb) | (s > ub)
            iqr_count = int(iqr_mask.sum())
            iqr_pct = self._pct(iqr_count, len(s))

            z_scores = np.abs(stats.zscore(s))
            z_mask = z_scores > self.ZSCORE_THRESHOLD
            z_count = int(z_mask.sum())
            z_pct = self._pct(z_count, len(s))

            iqr_passed = iqr_pct <= OUTLIER_THRESHOLD_PCT
            z_passed   = z_pct   <= OUTLIER_THRESHOLD_PCT

            col_results[col] = {
                "n_non_null": len(s),
                "iqr_method": {
                    "lower_fence": round(float(lb), 4),
                    "upper_fence": round(float(ub), 4),
                    "outlier_count": iqr_count,
                    "outlier_pct": iqr_pct,
                    "status": self._pass_fail(iqr_passed),
                },
                "zscore_method": {
                    "threshold": self.ZSCORE_THRESHOLD,
                    "outlier_count": z_count,
                    "outlier_pct": z_pct,
                    "status": self._pass_fail(z_passed),
                },
                "sample_outlier_values_iqr": sorted(
                    [round(float(x), 2) for x in s[iqr_mask].unique()[:10]]
                ),
                "overall_status": self._pass_fail(iqr_passed and z_passed),
            }

        total  = len(col_results)
        passed = sum(1 for v in col_results.values() if v["overall_status"] == "PASS")

        report = {
            "dimension": "Outlier Detection",
            "methods_used": ["IQR (Tukey fences, multiplier=1.5)", "Z-score (threshold=3.0)"],
            "iqr_multiplier": self.IQR_MULTIPLIER,
            "zscore_threshold": self.ZSCORE_THRESHOLD,
            "outlier_flag_threshold_pct": OUTLIER_THRESHOLD_PCT,
            "columns": col_results,
            "total_checks": total,
            "passed": passed,
            "failed": total - passed,
            "overall_status": self._pass_fail(passed == total),
        }

        self.results["5_outlier_detection"] = report
        logger.info(f"  Outlier Detection: {passed}/{total} columns within threshold")
        return report

    def check_distribution_profile(self) -> dict:
        logger.info("Running [6] Distribution Profile...")

        numeric_profile = {}
        for col in self.NUMERIC_COLS:
            if col not in self.df.columns:
                continue
            s = self.df[col].dropna()
            if len(s) == 0:
                continue
            lb, ub = self._iqr_bounds(s)
            iqr_outliers = int(((s < lb) | (s > ub)).sum())

            numeric_profile[col] = {
                "count": int(len(s)),
                "null_count": int(self.df[col].isna().sum()),
                "null_pct": self._pct(int(self.df[col].isna().sum()), self.n_rows),
                "mean": round(float(s.mean()), 4),
                "median": round(float(s.median()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "max": round(float(s.max()), 4),
                "q1":  round(float(s.quantile(0.25)), 4),
                "q3": round(float(s.quantile(0.75)), 4),
                "iqr": round(float(s.quantile(0.75) - s.quantile(0.25)), 4),
                "skewness": round(float(s.skew()), 4),
                "kurtosis": round(float(s.kurtosis()), 4),
                "distribution_shape": (
                    "heavily_right_skewed" if s.skew() > 2 else
                    "right_skewed" if s.skew() > 1 else
                    "left_skewed" if s.skew() < -1 else
                    "approximately_symmetric"
                ),
                "iqr_outlier_count": iqr_outliers,
                "iqr_outlier_pct": self._pct(iqr_outliers, len(s)),
                "iqr_lower_fence": round(float(lb), 4),
                "iqr_upper_fence": round(float(ub), 4),
            }

        
        categorical_profile = {}
        for col in self.CATEGORICAL_COLS:
            if col not in self.df.columns:
                continue
            vc = self.df[col].value_counts(dropna=False)
            top_val_pct = self._pct(int(vc.iloc[0]), self.n_rows) if len(vc) else 0
            categorical_profile[col] = {
                "unique_values":     int(self.df[col].nunique(dropna=True)),
                "null_count":        int(self.df[col].isna().sum()),
                "top_10_values":     {str(k): int(v) for k, v in vc.head(10).items()},
                "dominant_value_pct": top_val_pct,
                "balance_flag": (
                    "highly_imbalanced" if top_val_pct > 70 else
                    "imbalanced"        if top_val_pct > 50 else
                    "balanced"
                ),
            }

        sanity_checks = []

        if "review_scores_rating" in self.df.columns:
            s = self.df["review_scores_rating"].dropna()
            median_rating = float(s.median())
            high_pct = self._pct(int((s >= 3.5).sum()), len(s))
            sanity_checks.append({
                "check": "Median review_scores_rating in [4.0, 5.0]",
                "value": round(median_rating, 4),
                "status": self._pass_fail(4.0 <= median_rating <= 5.0),
                "note": "Airbnb ratings are known to skew positive",
            })
            sanity_checks.append({
                "check": ">= 85% of ratings in [3.5, 5.0]",
                "value_pct": high_pct,
                "status": self._pass_fail(high_pct >= 85.0),
            })

        if "price" in self.df.columns:
            median_price = float(self.df["price"].dropna().median())
            sanity_checks.append({
                "check": "Median price in [$50, $300]",
                "value": round(median_price, 2),
                "status": self._pass_fail(50 <= median_price <= 300),
            })

        if "room_type" in self.df.columns:
            most_common = self.df["room_type"].mode()[0]
            sanity_checks.append({
                "check": "Most common room_type is Entire home/apt or Private room",
                "value": most_common,
                "status": self._pass_fail(
                    most_common in {"Entire home/apt", "Private room"}
                ),
            })

        if "accommodates" in self.df.columns:
            mode_acc = int(self.df["accommodates"].mode()[0])
            sanity_checks.append({
                "check": "Most common accommodates value in [1, 4]",
                "value": mode_acc,
                "status": self._pass_fail(1 <= mode_acc <= 4),
                "note": "Most Airbnb listings are for 1-4 guests",
            })

        passed = sum(1 for c in sanity_checks if c["status"] == "PASS")
        report = {
            "dimension": "Distribution Profile",
            "numeric_profiles": numeric_profile,
            "categorical_profiles":  categorical_profile,
            "sanity_checks": sanity_checks,
            "sanity_checks_passed": passed,
            "sanity_checks_failed": len(sanity_checks) - passed,
            "overall_status": self._pass_fail(passed == len(sanity_checks)),
        }

        self.results["7_distribution_profile"] = report
        logger.info(
            f"  Distribution Profile: {passed}/{len(sanity_checks)} sanity checks passed"
        )
        return report

    def check_relationships(self) -> dict:
        logger.info("Running [7] Relationships...")

        numeric_df = self.df[
            [c for c in self.NUMERIC_COLS if c in self.df.columns]
        ].dropna()

        if len(numeric_df) == 0:
            report = {"dimension": "Relationships", "error": "No complete numeric rows"}
            self.results["8_relationships"] = report
            return report

        
        expected_positive_pairs = [
            ("bedrooms", "accommodates"),
            ("beds", "accommodates"),
            ("beds", "bedrooms"),
            ("price", "accommodates"),
            ("price", "bedrooms"),
            ("bathrooms", "bedrooms"),
        ]

        pair_results = []
        for col_a, col_b in expected_positive_pairs:
            if col_a not in numeric_df or col_b not in numeric_df:
                continue
            pearson_r,  pearson_p  = stats.pearsonr( numeric_df[col_a], numeric_df[col_b])
            spearman_r, spearman_p = stats.spearmanr(numeric_df[col_a], numeric_df[col_b])
            passed = pearson_r > 0
            pair_results.append({
                "pair": f"{col_a} vs {col_b}",
                "expected": "positive_correlation",
                "pearson_r":  round(float(pearson_r),  4),
                "pearson_p":  round(float(pearson_p),  6),
                "spearman_r": round(float(spearman_r), 4),
                "spearman_p": round(float(spearman_p), 6),
                "strength": (
                    "strong"   if abs(pearson_r) > 0.6 else
                    "moderate" if abs(pearson_r) > 0.3 else
                    "weak"
                ),
                "statistically_significant": bool(pearson_p < 0.05),
                "status": self._pass_fail(passed),
            })

        
        target = "review_scores_rating"
        target_correlations = {}
        if target in numeric_df.columns:
            feature_cols = [
                c for c in self.NUMERIC_COLS
                if c in numeric_df.columns and c != target
            ]
            for col in feature_cols:
                r,  p  = stats.pearsonr( numeric_df[col], numeric_df[target])
                sr, sp = stats.spearmanr(numeric_df[col], numeric_df[target])
                target_correlations[col] = {
                    "pearson_r":   round(float(r),  4),
                    "pearson_p":   round(float(p),  6),
                    "spearman_r":  round(float(sr), 4),
                    "spearman_p":  round(float(sp), 6),
                    "significant": bool(p < 0.05),
                    "direction":   "positive" if r > 0 else "negative",
                    "strength": (
                        "strong"   if abs(r) > 0.6 else
                        "moderate" if abs(r) > 0.3 else
                        "weak"
                    ),
                }
        
        corr_matrix = numeric_df.corr(method="pearson").round(4).to_dict()

        passed = sum(1 for p in pair_results if p["status"] == "PASS")
        report = {
            "dimension": "Relationships",
            "pairwise_checks": {
                "results": pair_results,
                "total":  len(pair_results),
                "passed": passed,
                "failed": len(pair_results) - passed,
            },
            "target_feature_correlations": target_correlations,
            "pearson_correlation_matrix":  corr_matrix,
            "overall_status": self._pass_fail(passed == len(pair_results)),
        }

        self.results["8_relationships"] = report
        logger.info(
            f"  Relationships: {passed}/{len(pair_results)} directional checks passed"
        )
        return report

    def run_all(self) -> dict:
        logger.info("=" * 60)
        logger.info("Starting full DataValidator pipeline")
        logger.info("=" * 60)

        self.check_accuracy()
        self.check_completeness()
        self.check_consistency()
        self.check_uniqueness()
        self.check_outliers()
        self.check_distribution_profile()
        self.check_relationships()

        dim_statuses = {
            k: v.get("overall_status", "N/A")
            for k, v in self.results.items()
        }
        all_passed = all(s == "PASS" for s in dim_statuses.values())

        report = {
            "metadata": {
                "generated_at":   datetime.now().isoformat(),
                "dataset_path":   self.dataset_path,
                "total_rows":     self.n_rows,
                "total_columns":  self.n_cols,
                "columns":        list(self.df.columns),
            },
            "summary": {
                "dimensions_evaluated": len(self.results),
                "dimensions_passed": sum(
                    1 for s in dim_statuses.values() if s == "PASS"
                ),
                "dimension_statuses": dim_statuses,
                "overall_status": "PASS" if all_passed else "FAIL",
            },
            "dimensions": self.results,
        }

        logger.info("")
        logger.info("=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        for dim, status in dim_statuses.items():
            icon = "+" if status == "PASS" else "x"
            logger.info(f"  [{icon}]  {dim:40s}  {status}")
        logger.info("-" * 60)
        logger.info(
            f"  Overall: {'PASS' if all_passed else 'FAIL'}  "
            f"({sum(1 for s in dim_statuses.values() if s == 'PASS')}/"
            f"{len(dim_statuses)} dimensions passed)"
        )
        logger.info("=" * 60)

        return report

if __name__ == "__main__":
    DATASET_PATH = "/home/abdallah/CMP4-Semster2/Airbnb-Rating-Classification/data/merged_airbnb_data.csv"

    logger.info(f"Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)

    validator = DataValidator(df, dataset_path=DATASET_PATH)
    report = validator.run_all()

    output_path = REPORT_DIR / "validation_report_full.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"\nFull report saved to: {output_path}")