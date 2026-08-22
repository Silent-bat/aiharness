class PaymentGateway:
    def capture(self, amount: int) -> str:
        if amount <= 0:
            raise ValueError("amount must be positive")
        return "payment-accepted"


def validate_amount(amount: int) -> None:
    if amount <= 0:
        raise ValueError("amount must be positive")

