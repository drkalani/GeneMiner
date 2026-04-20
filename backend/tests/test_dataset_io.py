"""Tests for article dataset parsing (DKDM-style columns, PMID dedupe)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class TestReadArticles(unittest.TestCase):
    def test_title_abstract_columns(self) -> None:
        from app.services.dataset_io import read_articles_from_path

        csv = (
            "pmid,title,abstract,label\n"
            "1,My title,Abstract body one,1\n"
            "2,Other,Second abstract,0\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(csv)
            p = Path(tmp.name)
        try:
            _df, rows = read_articles_from_path(p)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["pmid"], "1")
            self.assertEqual(rows[0]["title"], "My title")
            self.assertEqual(rows[0]["text"], "Abstract body one")
            self.assertEqual(rows[0]["label"], 1)
        finally:
            p.unlink(missing_ok=True)

    def test_dedupe_keeps_first(self) -> None:
        from app.services.dataset_io import read_articles_from_path

        csv = "pmid,text,label\n1,First,1\n1,Second ignored,0\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(csv)
            p = Path(tmp.name)
        try:
            _df, rows = read_articles_from_path(p)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["text"], "First")
            self.assertEqual(rows[0]["label"], 1)
        finally:
            p.unlink(missing_ok=True)


class TestTrainingHelpers(unittest.TestCase):
    def test_dedupe_articles(self) -> None:
        from app.schemas.models import ArticleInput
        from app.services.training_runner import _articles_to_training, _dedupe_articles

        arts = [
            ArticleInput(pmid="1", text="a", label=1, title=None),
            ArticleInput(pmid="1", text="b", label=0, title=None),
        ]
        d = _dedupe_articles(arts)
        self.assertEqual(len(d), 1)
        self.assertEqual(d[0].text, "a")

    def test_pair_mode_when_any_title(self) -> None:
        from app.schemas.models import ArticleInput
        from app.services.training_runner import _articles_to_training

        arts = [
            ArticleInput(pmid="1", text="abstract one", label=1, title="T1"),
            ArticleInput(pmid="2", text="abstract two", label=0, title=""),
        ]
        texts, labels, _pmids, titles_b = _articles_to_training(arts)
        self.assertIsNotNone(titles_b)
        assert titles_b is not None
        self.assertEqual(titles_b[0], "T1")
        self.assertEqual(titles_b[1], "")


if __name__ == "__main__":
    unittest.main()
