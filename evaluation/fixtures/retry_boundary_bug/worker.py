from retry_policy import permits_another_attempt


def execute_job(attempts):
    return permits_another_attempt(attempts)
