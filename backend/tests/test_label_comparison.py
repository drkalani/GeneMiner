"""Label comparison (LitSuggest-style) tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.services.label_comparison import compare_pmids_labels_vs_scores


class TestCompare(unittest.TestCase):
    def test_basic_agreement(self) -> None:
        out = compare_pmids_labels_vs_scores(
            [{"pmid": "1", "label": 1}, {"pmid": "2", "label": 0}],
            [{"pmid": "1", "score": 0.9}, {"pmid": "2", "score": 0.1}],
            score_threshold=0.5,
        )
        self.assertEqual(out["intersection_count"], 2)
        self.assertEqual(out["mismatch_count"], 0)
        self.assertEqual(out["metrics"]["accuracy"], 1.0)

    def test_mismatch(self) -> None:
        out = compare_pmids_labels_vs_scores(
            [{"pmid": "1", "relevant": 0}],
            [{"pmid": "1", "score": 0.9}],
            score_threshold=0.5,
        )
        self.assertEqual(out["mismatch_count"], 1)
        self.assertEqual(out["agreement_count"], 0)


class TestLitSuggestCsv(unittest.TestCase):
    def test_read_scores(self) -> None:
        from app.services.dataset_io import read_litsuggest_scores_from_path

        csv = "pmid,score\n10,0.7\n11,0.2\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(csv)
            p = Path(tmp.name)
        try:
            _df, rows = read_litsuggest_scores_from_path(p)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["pmid"], "10")
            self.assertAlmostEqual(rows[0]["score"], 0.7)
        finally:
            p.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
