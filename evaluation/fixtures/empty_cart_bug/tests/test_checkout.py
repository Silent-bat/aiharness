import unittest

from checkout import submit_order
from payments import PaymentGateway


class CheckoutTest(unittest.TestCase):
    def test_empty_cart_is_rejected(self):
        with self.assertRaises(ValueError):
            submit_order([], PaymentGateway())
