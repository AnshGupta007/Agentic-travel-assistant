"""Pytest configuration and environment compatibility hooks."""

import sys

# Workaround for Starlette v0.40+ compatibility with Streamlit starlette gzip middleware
try:
    import starlette.middleware.gzip
    if not hasattr(starlette.middleware.gzip, "DEFAULT_EXCLUDED_CONTENT_TYPES"):
        starlette.middleware.gzip.DEFAULT_EXCLUDED_CONTENT_TYPES = (
            "text/html",
            "text/css",
            "text/plain",
            "application/javascript",
            "application/json",
        )
except ImportError:
    pass
