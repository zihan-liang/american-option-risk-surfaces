import csv
import unittest
from pathlib import Path

from common import write_csv


class CommonIoTests(unittest.TestCase):
    def test_write_csv_accepts_rows_with_case_specific_columns(self):
        path = Path("results/logs/test_common_io_rows.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(path, [{"a": 1}, {"a": 2, "b": 3}])

        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))

        self.assertEqual(rows[0]["a"], "1")
        self.assertEqual(rows[1]["b"], "3")

    def test_write_csv_uses_lf_line_endings(self):
        path = Path("results/logs/test_common_io_line_endings.csv")
        path.parent.mkdir(parents=True, exist_ok=True)

        write_csv(path, [{"a": 1}, {"a": 2}])

        self.assertNotIn(b"\r\n", path.read_bytes())
