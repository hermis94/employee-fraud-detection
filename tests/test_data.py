import unittest

from src.data import FEATURES, generate_dataset


class ExpenseDataTests(unittest.TestCase):
    def test_generation_is_reproducible_and_unlabelled(self):
        first = generate_dataset(rows=300)
        second = generate_dataset(rows=300)
        self.assertEqual(first.to_csv(index=False), second.to_csv(index=False))
        self.assertTrue(set(FEATURES).issubset(first.columns))
        self.assertNotIn("is_fraud", first.columns)
        self.assertTrue(first["claim_id"].is_unique)


if __name__ == "__main__":
    unittest.main()