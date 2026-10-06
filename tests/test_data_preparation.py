import json
from pathlib import Path
import tempfile
import unittest

from data.prepare_data import split_interactions
from data.sentires import SentiresGuideExtractor


class TemporalPreparationTest(unittest.TestCase):
    def test_history_and_positive_features_exclude_future_interactions(self):
        records = [
            {"user_id": "u", "item_id": 1, "timestamp": 1, "text": "warm plot"},
            {"user_id": "u", "item_id": 2, "timestamp": 2, "text": "dark ending"},
            {"user_id": "u", "item_id": 3, "timestamp": 3, "text": "future review"},
            {"user_id": "v", "item_id": 4, "timestamp": 1, "text": "warm plot"},
            {"user_id": "v", "item_id": 5, "timestamp": 2, "text": "other second"},
            {"user_id": "v", "item_id": 6, "timestamp": 3, "text": "other third"},
        ]
        sentires_rows = [
            {"text": "warm plot", "sentence": [["plot", "warm", "warm plot", 1.0]]},
            {"text": "dark ending", "sentence": [["ending", "dark", "dark ending", -1.0]]},
            {"text": "future review", "sentence": [["future", "good", "future review", 1.0]]},
            {"text": "warm plot", "sentence": [["noise", "warm", "warm plot", 1.0]]},
            {"text": "other second", "sentence": []},
            {"text": "other third", "sentence": []},
        ]
        for source_record, sentires_record in zip(records, sentires_rows):
            sentires_record["user"] = source_record["user_id"]
            sentires_record["item"] = source_record["item_id"]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sentires.json"
            source.write_text(json.dumps(sentires_rows), encoding="utf-8")
            extractor = SentiresGuideExtractor(str(source))
            splits = split_interactions(
                records, {str(i): [] for i in range(1, 8)}, extractor,
                history_len=10, candidate_count=5, seed=7,
            )

        validation = next(row for row in splits["valid"] if row["user_id"] == "u")
        test = next(row for row in splits["test"] if row["user_id"] == "u")
        self.assertEqual(validation["history"], [1])
        self.assertEqual(test["history"], [1, 2])
        self.assertEqual(test["user_positive_features"], ["plot"])
        self.assertNotIn("future", test["user_positive_features"])
        self.assertNotIn("noise", test["user_positive_features"])
        self.assertTrue(set(test["negative_items"]).isdisjoint({1, 2, 3}))
        self.assertIn(7, test["negative_items"])


if __name__ == "__main__":
    unittest.main()
