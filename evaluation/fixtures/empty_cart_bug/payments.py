class PaymentGateway:
    def capture(self, amount: int) -> str:
        return "payment-accepted"


def validate_amount(amount: int) -> None:
    # Deliberate benchmark bug: zero should be rejected as well.
    if amount < 0:
        raise ValueError("amount must be positive")
