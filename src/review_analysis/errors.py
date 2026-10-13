"""Domain errors. The API layer maps each one to an HTTP status."""


class ReviewAnalysisError(Exception):
    """Base class for all expected, user-facing errors."""


class InvalidInputError(ReviewAnalysisError):
    """The request is malformed: a non-numeric app ID, a bad storefront code, etc. (422)."""


class AppNotFoundError(ReviewAnalysisError):
    """The app does not exist, or it is not sold in the requested storefront (404)."""


class UpstreamError(ReviewAnalysisError):
    """Apple could not be reached or kept failing after retries (502/503)."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class BudgetExceededError(UpstreamError):
    """The run's overall time budget ran out, usually because of sustained rate limiting."""
