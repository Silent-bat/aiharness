from validation import sanitize_payload


def process_request(payload):
    return sanitize_payload(payload)
