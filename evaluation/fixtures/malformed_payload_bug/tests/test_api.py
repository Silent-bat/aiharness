import unittest

from api import process_request


class ApiTest(unittest.TestCase):
    def test_malformed_input_is_rejected(self):
        with self.assertRaises(ValueError):
            process_request("not-a-mapping")
