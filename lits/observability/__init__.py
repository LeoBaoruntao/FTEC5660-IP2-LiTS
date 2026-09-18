"""Optional LiTS tracing with local archives and configurable observations.

Import with ``from lits import observability as tracing``. See
``lits.observability.tracing`` for configuration examples and archive contents.
"""

from .tracing import attach_logging_handler, close, configure, flush, instrument_openai, observe, span
from .archive import filter_archives, upload_archives

__all__ = ["configure", "span", "observe", "instrument_openai", "attach_logging_handler", "flush", "close",
           "filter_archives", "upload_archives"]
