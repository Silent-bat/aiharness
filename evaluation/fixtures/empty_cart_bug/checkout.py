from payments import PaymentGateway, validate_amount


def checkout_total(items):
    return sum(item["price"] for item in items)


def submit_order(items, gateway: PaymentGateway):
    total = checkout_total(items)
    validate_amount(total)
    return gateway.capture(total)
