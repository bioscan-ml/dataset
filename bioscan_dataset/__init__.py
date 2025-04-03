from . import __meta__

__version__ = __meta__.version

__all__ = ["BIOSCAN1M", "BIOSCAN5M", "load_bioscan1m_metadata", "load_bioscan5m_metadata"]

from .bioscan1m import BIOSCAN1M, load_bioscan1m_metadata
from .bioscan5m import BIOSCAN5M, load_bioscan5m_metadata
