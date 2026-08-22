def sanitize_payload(payload):
    return {key: value.strip() for key, value in payload.items()}
