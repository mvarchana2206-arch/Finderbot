#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------


def raise_verify_error(error: Exception) -> None:
    """
    Raise OSError with detailed message for SSL verification errors.

    Args:
        error: The original exception
    """
    raise OSError(
        f"Connection cannot be verified with default trusted CAs. "
        f"Please provide correct path to a CA_BUNDLE file or directory with "
        f"certificates of trusted CAs. Error: {error}"
    ) from error
