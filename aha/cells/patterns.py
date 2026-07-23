"""Temporal pattern detection for LPCCs.

BIOLOGICAL MOTIVATION:
  Cortical cells respond to temporal signatures, not isolated stimuli.
IMPLEMENTED MECHANISM:
  A bounded, symbolic temporal signature built from the recent context window.
  A prototype pattern is a set of (feature, value) tokens plus optional recent
  action/reward tokens. Resonance = fraction of prototype tokens present in the
  current temporal signature, discounted by a mismatch penalty.
UNRESOLVED BIOLOGICAL QUESTION:
  How real dendritic sequence detectors integrate over time is out of scope.

The detector is deliberately replaceable (see PatternDetector protocol) and uses
NO neural embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


Token = tuple[str, str]  # (namespace_key, value_repr)


def tokenize_observation(obs: dict[str, Any], prefix: str = "obs") -> set[Token]:
    """Convert a symbolic observation dict into a set of tokens."""
    tokens: set[Token] = set()
    for key, value in obs.items():
        tokens.add((f"{prefix}:{key}", str(value)))
    return tokens


@dataclass
class TemporalSignature:
    """A bounded, inspectable temporal fingerprint of recent experience."""

    tokens: set[Token]
    horizon_seen: int

    def __contains__(self, token: Token) -> bool:
        return token in self.tokens


class PatternDetector(Protocol):
    """Replaceable pattern-detection interface. No neural embedding allowed."""

    def signature(
        self,
        current_obs: dict[str, Any],
        recent_obs: list[dict[str, Any]],
        recent_actions: list[str],
        recent_reward: float,
        self_prev_activation: float,
    ) -> TemporalSignature: ...

    def resonance(self, prototype: set[Token], signature: TemporalSignature) -> float: ...


@dataclass
class SymbolicTemporalDetector:
    """Default bounded symbolic temporal detector.

    Builds a token set from:
      - current observation,
      - up to `window` recent observations (time-tagged coarsely),
      - the most recent action,
      - a coarse reward-sign token,
      - a coarse self-activation token.
    """

    window: int = 3
    reward_threshold: float = 0.01

    def signature(
        self,
        current_obs: dict[str, Any],
        recent_obs: list[dict[str, Any]],
        recent_actions: list[str],
        recent_reward: float,
        self_prev_activation: float,
    ) -> TemporalSignature:
        tokens: set[Token] = set()
        tokens |= tokenize_observation(current_obs, prefix="now")
        # Recent observations get a coarse relative-time tag (t-1, t-2, ...).
        for lag, obs in enumerate(reversed(recent_obs[-self.window :]), start=1):
            tokens |= tokenize_observation(obs, prefix=f"t-{lag}")
        if recent_actions:
            tokens.add(("act:last", str(recent_actions[-1])))
        if recent_reward > self.reward_threshold:
            tokens.add(("rew:sign", "pos"))
        elif recent_reward < -self.reward_threshold:
            tokens.add(("rew:sign", "neg"))
        else:
            tokens.add(("rew:sign", "zero"))
        tokens.add(("self:active", "1" if self_prev_activation >= 0.5 else "0"))
        return TemporalSignature(tokens=tokens, horizon_seen=len(recent_obs))

    def resonance(self, prototype: set[Token], signature: TemporalSignature) -> float:
        """Fraction of prototype tokens present in the signature.

        Empty prototype => zero resonance (an undefined cell cannot resonate).
        """
        if not prototype:
            return 0.0
        hits = sum(1 for token in prototype if token in signature)
        return hits / len(prototype)