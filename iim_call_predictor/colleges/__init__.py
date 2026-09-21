"""Importing this package registers every available college plug-in.

To add a new IIM: create a new subpackage here and import it below so its
``@register_college(...)`` decorator runs.
"""

from . import iima  # noqa: F401  (import triggers registration)
from . import iimm  # noqa: F401  (import triggers registration)
from . import iimc  # noqa: F401  (import triggers registration)
from . import iimb  # noqa: F401  (import triggers registration)
from . import iiml  # noqa: F401  (import triggers registration)
