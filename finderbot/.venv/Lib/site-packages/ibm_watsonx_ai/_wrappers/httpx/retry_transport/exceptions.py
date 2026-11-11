#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------


class NoneResponseError(ValueError):
    """
    The value returned from `HTTPTransport.handle_request` or
    `AsyncHTTPTransport.handle_async_request` is None. This exception
    may appear when `instana<3.15` package is installed on the environment.
    This has first appeared in CPD 5.4 and this workaround can be removed
    only when CPD 5.4 stops receiving Python SDK support.

    Related issue: https://github.com/instana/python-sensor/issues/866

    (To whoever is removing this in the future: Have a nice day :3)
    """
