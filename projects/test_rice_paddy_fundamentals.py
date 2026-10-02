import copy
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from rice_paddy_fundamentals import (
    build_features,
    feature_columns,
    load_bundle,
    validate_bundle,
)


def base_bundle():
    """Hand-checked public-data-shaped fixture; all dates include an offset."""
    return {
        "schema_version": 1,
        "series": [
            {
                "id": "mp_arrivals",
                "block": "supply",
                "unit": "tonnes",
                "geography": "Madhya Pradesh",
                "commodity": "paddy",
                "description": "Monthly contextual arrivals",
                "kind": "observation",
                "max_age_days": 120,
            },
            {
                "id": "procurement_state",
                "block": "policy",
                "unit": "state_code",
                "geography": "India",
                "commodity": "paddy",
                "description": "Contextual procurement-policy state",
                "kind": "state",
                "max_age_days": None,
            },
        ],
        "records": [
            {
                "id": "arrivals-jan-initial",
                "series_id": "mp_arrivals",
                "period_start": "2024-01-01",
                "period_end": "2024-01-31",
                "value": 10,
                "available_at": "2024-02-10T09:00:00+05:30",
                "review_status": "verified",
                "source_document_id": "doc-arrivals",
                "source_url": "https://example.test/arrivals/initial",
            },
            {
                "id": "arrivals-jan-revision",
                "series_id": "mp_arrivals",
                "period_start": "2024-01-01",
                "period_end": "2024-01-31",
                "value": 99,
                "available_at": "2024-04-10T09:00:00+05:30",
                "review_status": "verified",
                "source_document_id": "doc-arrivals-revised",
                "source_url": "https://example.test/arrivals/revised",
            },
            {
                "id": "policy-prior",
                "series_id": "procurement_state",
                "period_start": "2024-01-01",
                "period_end": "2024-01-31",
                "value": 1,
                "available_at": "2024-01-01T09:00:00+05:30",
                "effective_at": "2024-01-01T00:00:00+05:30",
                "review_status": "verified",
                "source_document_id": "doc-policy-prior",
                "source_url": "https://example.test/policy/prior",
            },
            {
                "id": "policy-august",
                "series_id": "procurement_state",
                "period_start": "2024-08-01",
                "period_end": "2024-08-31",
                "value": 2,
                "available_at": "2024-07-10T09:00:00+05:30",
                "effective_at": "2024-08-01T00:00:00+05:30",
                "review_status": "verified",
                "source_document_id": "doc-policy-august",
                "source_url": "https://example.test/policy/august",
            },
            {
                "id": "policy-october-repeal",
                "series_id": "procurement_state",
                "period_start": "2024-10-01",
                "period_end": "2024-10-31",
                "value": 0,
                "available_at": "2024-10-05T09:00:00+05:30",
                "effective_at": "2024-10-01T00:00:00+05:30",
                "review_status": "verified",
                "source_document_id": "doc-policy-repeal",
                "source_url": "https://example.test/policy/repeal",
            },
        ],
        "documents": [
            {
                "id": document_id,
                "url": {
                    "doc-arrivals": "https://example.test/arrivals/initial",
                    "doc-arrivals-revised": "https://example.test/arrivals/revised",
                    "doc-policy-prior": "https://example.test/policy/prior",
                    "doc-policy-august": "https://example.test/policy/august",
                    "doc-policy-repeal": "https://example.test/policy/repeal",
                }[document_id],
                "published_at": {
                    "doc-arrivals": "2024-02-10T09:00:00+05:30",
                    "doc-arrivals-revised": "2024-04-10T09:00:00+05:30",
                    "doc-policy-prior": "2024-01-01T09:00:00+05:30",
                    "doc-policy-august": "2024-07-10T09:00:00+05:30",
                    "doc-policy-repeal": "2024-10-05T09:00:00+05:30",
                }[document_id],
                "publication_evidence": "fixture",
                "local_path": None,
                "sha256": None,
                "access_status": "inspected",
                "notes": "synthetic fixture",
            }
            for document_id in (
                "doc-arrivals", "doc-arrivals-revised", "doc-policy-prior",
                "doc-policy-august", "doc-policy-repeal",
            )
        ],
    }


class FundamentalsTests(unittest.TestCase):
    def examples(self):
        return pd.DataFrame(
            {
                "origin": ["2024-03-01", "2024-05-01", "2024-07-31", "2024-09-30"],
                "target_month": ["2024-04-30", "2024-06-30", "2024-09-30", "2024-10-31"],
            },
            index=pd.Index(["march", "may", "july", "september"], name="case"),
        )

    def test_observations_are_selected_as_of_origin_not_latest_retrieval(self):
        """Would fail if a future revision leaks into an earlier origin."""
        features, provenance = build_features(self.examples(), base_bundle())
        col = "exog_supply_mp_arrivals"
        self.assertEqual(features.loc["march", col], 10)
        self.assertEqual(features.loc["may", col], 99)
        self.assertEqual(
            provenance.loc[provenance.origin.eq(pd.Timestamp("2024-03-01")), "record_ids"].iloc[0],
            ["arrivals-jan-initial"],
        )

    def test_features_preserve_index_order_and_declare_only_requested_blocks(self):
        """Would fail if joins reorder cases or feature allowlists include policy."""
        features, _ = build_features(self.examples(), base_bundle())
        self.assertEqual(features.index.tolist(), ["march", "may", "july", "september"])
        self.assertEqual(
            feature_columns(base_bundle(), ("supply",)),
            ["exog_supply_mp_arrivals", "exog_supply_mp_arrivals_missing", "exog_supply_mp_arrivals_age_days"],
        )
        self.assertIn("exog_policy_procurement_state_known_target", features.columns)

    def test_features_preserve_duplicate_example_index_values_without_cross_writing(self):
        """Would fail if label-based assignment overwrites another same-named case."""
        examples = self.examples().iloc[:2].copy()
        examples.index = pd.Index(["same-case", "same-case"], name="case")
        features, _ = build_features(examples, base_bundle())
        self.assertEqual(features.index.tolist(), ["same-case", "same-case"])
        self.assertEqual(features.iloc[0]["exog_supply_mp_arrivals"], 10)
        self.assertEqual(features.iloc[1]["exog_supply_mp_arrivals"], 99)

    def test_pending_or_undated_records_are_missing_not_zero(self):
        """Would fail if unreviewed or undated evidence becomes a usable value."""
        bundle = base_bundle()
        bundle["records"] = [
            {**bundle["records"][0], "id": "pending", "value": 7, "review_status": "pending"},
            {**bundle["records"][0], "id": "undated", "value": 8, "available_at": None},
        ]
        features, provenance = build_features(self.examples().iloc[:1], bundle)
        self.assertTrue(math.isnan(features.iloc[0]["exog_supply_mp_arrivals"]))
        self.assertEqual(features.iloc[0]["exog_supply_mp_arrivals_missing"], 1)
        self.assertTrue(math.isnan(features.iloc[0]["exog_supply_mp_arrivals_age_days"]))
        self.assertEqual(provenance.iloc[0]["status"], "missing")

    def test_zero_is_valid_and_stale_observation_is_suppressed(self):
        """Would fail if zero is conflated with missing or max age is ignored."""
        bundle = base_bundle()
        bundle["series"][0]["max_age_days"] = 60
        bundle["records"] = [{**bundle["records"][0], "id": "zero", "value": 0}]
        features, _ = build_features(self.examples().iloc[:1], bundle)
        self.assertEqual(features.iloc[0]["exog_supply_mp_arrivals"], 0)
        self.assertEqual(features.iloc[0]["exog_supply_mp_arrivals_missing"], 0)
        stale, provenance = build_features(self.examples().iloc[[1]], bundle)
        self.assertTrue(math.isnan(stale.iloc[0]["exog_supply_mp_arrivals"]))
        self.assertEqual(stale.iloc[0]["exog_supply_mp_arrivals_missing"], 1)
        self.assertEqual(provenance.iloc[0]["status"], "stale")

    def test_policy_current_and_known_target_are_as_of_announcements(self):
        """Would fail if future effective/repeal states are read at a July origin."""
        features, _ = build_features(self.examples(), base_bundle())
        current = "exog_policy_procurement_state"
        known_target = "exog_policy_procurement_state_known_target"
        self.assertEqual(features.loc["july", current], 1)
        self.assertEqual(features.loc["july", known_target], 2)
        self.assertEqual(features.loc["september", current], 2)
        self.assertEqual(features.loc["september", known_target], 2)

    def test_state_age_is_days_since_the_selected_effective_state_at_origin(self):
        """Would fail if state age is discarded instead of exposing policy duration."""
        features, _ = build_features(self.examples(), base_bundle())
        age = "exog_policy_procurement_state_age_days"
        self.assertEqual(features.loc["july", age], 212)
        self.assertEqual(features.loc["september", age], 60)

    def test_known_target_provenance_remains_auditable_when_current_state_is_missing(self):
        """Would fail if a future-effective known target loses its source audit fields."""
        bundle = base_bundle()
        bundle["records"] = [
            record for record in bundle["records"]
            if record["id"] in {"policy-august", "policy-october-repeal"}
        ]
        features, provenance = build_features(self.examples().loc[["july"]], bundle)
        current = "exog_policy_procurement_state"
        known_target = "exog_policy_procurement_state_known_target"
        self.assertTrue(math.isnan(features.loc["july", current]))
        self.assertEqual(features.loc["july", known_target], 2)
        row = provenance.loc[provenance.series_id.eq("procurement_state")].iloc[0]
        self.assertEqual(row["status"], "missing")
        self.assertEqual(row["known_target_status"], "selected")
        self.assertEqual(row["known_target_source_url"], "https://example.test/policy/august")
        self.assertEqual(row["known_target_available_at"], "2024-07-10T09:00:00+05:30")
        self.assertEqual(row["known_target_period_end"], "2024-08-31")

    def test_validation_rejects_series_that_collide_after_feature_name_generation(self):
        """Would fail if valid identifiers overwrite another series' missing flag."""
        bundle = base_bundle()
        bundle["series"].append(
            {
                "id": "mp_arrivals_missing",
                "block": "supply",
                "unit": "tonnes",
                "geography": "Madhya Pradesh",
                "commodity": "paddy",
                "description": "Deliberately colliding fixture",
                "kind": "observation",
                "max_age_days": 30,
            }
        )
        with self.assertRaisesRegex(ValueError, "generated feature column collision"):
            validate_bundle(bundle)

    def test_validation_rejects_invalid_schema_dates_values_and_duplicate_ids(self):
        """Would fail if malformed or ambiguous evidence is accepted."""
        cases = []
        naive = base_bundle()
        naive["records"][0]["available_at"] = "2024-02-10T09:00:00"
        cases.append(naive)
        naive_effective = base_bundle()
        naive_effective["records"][2]["effective_at"] = "2024-01-01T00:00:00"
        cases.append(naive_effective)
        unknown_reference_date = base_bundle()
        unknown_reference_date["records"][0]["period_end"] = "unknown"
        cases.append(unknown_reference_date)
        noncanonical_reference_date = base_bundle()
        noncanonical_reference_date["records"][0]["period_end"] = "20240131"
        cases.append(noncanonical_reference_date)
        invalid_unit = base_bundle()
        invalid_unit["series"][0]["unit"] = ""
        cases.append(invalid_unit)
        invalid_value = base_bundle()
        invalid_value["records"][0]["value"] = float("nan")
        cases.append(invalid_value)
        duplicate = base_bundle()
        duplicate["records"].append(copy.deepcopy(duplicate["records"][0]))
        cases.append(duplicate)
        bad_metadata = base_bundle()
        bad_metadata["series"][0]["id"] = "UpperCase"
        cases.append(bad_metadata)
        before_document_publication = base_bundle()
        before_document_publication["documents"][0]["published_at"] = "2024-02-11T09:00:00+05:30"
        cases.append(before_document_publication)
        mismatched_source_url = base_bundle()
        mismatched_source_url["records"][0]["source_url"] = "https://example.test/not-the-document"
        cases.append(mismatched_source_url)
        for bundle in cases:
            with self.subTest(bundle=bundle):
                with self.assertRaises(ValueError):
                    validate_bundle(bundle)

    def test_date_only_document_requires_next_day_record_availability(self):
        """Would fail if an imprecise publication date permits same-day leakage."""
        bundle = base_bundle()
        bundle["documents"][0]["published_at"] = "2024-02-10"
        with self.assertRaises(ValueError):
            validate_bundle(bundle)
        bundle["records"][0]["available_at"] = "2024-02-11T00:00:00+05:30"
        validate_bundle(bundle)

    def test_build_rejects_conflicting_equally_available_vintages(self):
        """Would fail if an arbitrary value wins a conflicting tie."""
        bundle = base_bundle()
        tied = {**bundle["records"][0], "id": "conflicting", "value": 11}
        bundle["records"] = [bundle["records"][0], tied]
        with self.assertRaises(ValueError):
            build_features(self.examples().iloc[:1], bundle)

    def test_empty_records_and_required_example_columns_are_handled_explicitly(self):
        """Would fail if no evidence crashes or required timing columns are ignored."""
        bundle = base_bundle()
        bundle["records"] = []
        features, provenance = build_features(self.examples().iloc[:1], bundle)
        self.assertTrue(features.iloc[0].filter(like="exog_").isna().any())
        self.assertEqual(len(provenance), 2)
        with self.assertRaises(ValueError):
            build_features(pd.DataFrame({"origin": ["2024-01-01"]}), bundle)

    def test_load_bundle_validates_json_before_returning_it(self):
        """Would fail if invalid persisted evidence can bypass schema validation."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.json"
            path.write_text('{"schema_version": 1, "series": [], "records": [], "documents": []}', encoding="utf-8")
            self.assertEqual(load_bundle(path)["schema_version"], 1)
            path.write_text('{"schema_version": 2}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_bundle(path)


if __name__ == "__main__":
    unittest.main()
