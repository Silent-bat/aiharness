from policy import has_required_scope


def authorize_request(scopes):
    if not has_required_scope(scopes):
        raise PermissionError("insufficient scope")
    return True
