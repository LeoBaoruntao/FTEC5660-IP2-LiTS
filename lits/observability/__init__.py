"""Optional LiTS tracing with local archives.

Import with ``from lits import observability as tracing``. See
``lits.observability.tracing`` for configuration examples and archive contents.
"""

from .tracing import close, configure, flush, span

__all__ = ["configure", "span", "flush", "close"]
