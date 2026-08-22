import unittest

from access import authorize_request


class AccessTest(unittest.TestCase):
    def test_write_only_credentials_are_rejected(self):
        with self.assertRaises(PermissionError):
            authorize_request(["write"])

    def test_admin_credentials_are_accepted(self):
        self.assertTrue(authorize_request(["admin"]))
