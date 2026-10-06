import unittest

from data.kg import KGTriple, KnowledgeGraph


class TargetPathTest(unittest.TestCase):
    def test_offline_paths_preserve_inverse_relation_direction(self):
        kg = KnowledgeGraph(
            {"history": 0, "bridge": 1, "target": 2},
            {"likes": 0, "has": 1},
            [KGTriple(0, 0, 1), KGTriple(2, 1, 1)],
        )
        paths = [
            record["path"]
            for record in kg.paths_around_target(2, max_hop=2)
        ]
        self.assertIn([(0, 0, 1), (1, 3, 2)], paths)
        self.assertEqual(kg.id2relation[3], "inverse_of_has")
        self.assertTrue(all(len(path) <= 2 for path in paths))


if __name__ == "__main__":
    unittest.main()
