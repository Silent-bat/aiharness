import unittest

from billing import charge_invoice


class BillingTest(unittest.TestCase):
    def test_decimal_price_uses_exact_cents(self):
        self.assertEqual(charge_invoice(1.15), 115)
