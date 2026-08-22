from checkout import submit_order
from payments import PaymentGateway


def test_empty_cart_is_rejected():
    try:
        submit_order([], PaymentGateway())
    except ValueError:
        return
    raise AssertionError("empty cart should be rejected")

