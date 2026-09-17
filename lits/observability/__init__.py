"""Optional LiTS tracing with local archives and configurable observations.

Import with ``from lits import observability as tracing``. See
``lits.observability.tracing`` for configuration examples and archive contents.
"""

from .tracing import close, configure, flush, observe, span

__all__ = ["configure", "span", "observe", "flush", "close"]
