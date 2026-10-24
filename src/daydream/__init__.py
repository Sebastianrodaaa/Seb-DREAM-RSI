"""Seb-DREAM-RSI: an independent implementation of Dream-RSI.

Dream-RSI (Zheng et al., 2026) improves an exploration policy by replaying
completed discovery trees instead of rerunning the discovery agent. This
package implements that loop. It is not the authors' unreleased codebase
and is not affiliated with them.
"""

from seb_dream_rsi.loop import RunResult, run_discovery
from seb_dream_rsi.replay import ReplayResult, replay, replay_score

__all__ = [
    "ReplayResult",
    "RunResult",
    "replay",
    "replay_score",
    "run_discovery",
]

__version__ = "0.1.0"
