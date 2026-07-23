"""Candidate hypothesis generation from experience.

BIOLOGICAL MOTIVATION:
  New predictive units recruit when the current population fails to explain a
  recurring temporal transition (structural plasticity / neurogenesis-like).

IMPLEMENTED MECHANISM:
  A lightweight transition miner observes (context signature -> next observation)
  pairs. When a specific antecedent token co-occurs repeatedly with a consequent
  token that no existing cell predicts, it proposes a new LocalHypothesis of the
  form C x A -> F. Splits (specialization) copy a documented subset of the
  parent context and narrow it with the most discriminative extra token.

No neural embedding, no gradient. Purely count-based, inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

from aha.cells.patterns import Token
from aha.hypotheses.hypothesis import LocalHypothesis


@dataclass
class _Antecedent:
    tokens: set[Token]
    action: str | None
    age: int = 0


@dataclass
class TransitionMiner:
    """Delay-aware action-conditioned transition miner.

    For each step it buffers the current antecedent state tokens + the action
    taken. It then attributes newly-arrived consequent tokens (up to max_delay
    steps later) to those buffered (antecedent, action) pairs. This lets it
    discover DELAYED, ACTION-CONDITIONED rules such as
        switch=present + investigate -> (delay) door=open
    which single-step co-occurrence mining cannot capture.

    Counts key: (antecedent_token, action, consequent_token, delay) -> count.
    Confidence proxy = count(ant,action->con) / count(ant,action) so spurious
    high-frequency consequents are down-weighted.
    """

    min_support: int = 3
    max_delay: int = 4
    counts: dict[tuple[Token, str | None, Token, int], int] = field(default_factory=dict)
    antecedent_counts: dict[tuple[Token, str | None], int] = field(default_factory=dict)
    _buffer: list[_Antecedent] = field(default_factory=list)
    _id_gen = count(1)

    def observe(
        self,
        prev_tokens: set[Token],
        action: str | None,
        next_tokens: set[Token],
    ) -> None:
        """Record buffered (antecedent, action) -> delayed consequent arrivals.

        Consequents include both local observation tokens (``now:``) AND the
        reward-sign token (``rew:sign``). Allowing reward as a minable consequent
        is what lets a downstream, partially-observing cell form a hypothesis of
        the form "received-message-token + my-signal + my-action -> reward",
        i.e. a COMMUNICATION-CONDITIONED reward predictor. Without this, a cell
        whose only observable downstream signal is reward can never attach a
        predictive consequent to a received message.
        """
        def minable(t: Token) -> bool:
            return t[0].startswith("now:") or t[0] == "rew:sign"

        prev_now = {t for t in prev_tokens if t[0].startswith("now:")}
        # Register this step's antecedent.
        self._buffer.append(_Antecedent(tokens=prev_now, action=action, age=0))
        for ant_tok in prev_now:
            key = (ant_tok, action)
            self.antecedent_counts[key] = self.antecedent_counts.get(key, 0) + 1

        # Newly-arrived consequent tokens this step (local obs OR positive reward).
        arrived = {t for t in next_tokens if minable(t)} - {t for t in prev_tokens if minable(t)}
        # Always allow a positive reward token to count as an arrival.
        arrived |= {t for t in next_tokens if t == ("rew:sign", "pos")}

        # Attribute arrivals to buffered antecedents at their current age (delay).
        for entry in self._buffer:
            delay = entry.age
            if delay < 1 or delay > self.max_delay:
                continue
            for ant_tok in entry.tokens:
                for con in arrived:
                    if con in entry.tokens:
                        continue
                    key = (ant_tok, entry.action, con, delay)
                    self.counts[key] = self.counts.get(key, 0) + 1

        # Age and prune the buffer.
        for entry in self._buffer:
            entry.age += 1
        self._buffer = [e for e in self._buffer if e.age <= self.max_delay]

    def propose(self, existing_signatures: set[frozenset[Token]]) -> list[LocalHypothesis]:
        """Emit hypotheses for well-supported, reasonably-precise transitions."""
        proposals: list[LocalHypothesis] = []
        for (ant, action, con, delay), n in self.counts.items():
            if n < self.min_support:
                continue
            base = self.antecedent_counts.get((ant, action), 0)
            precision = n / base if base else 0.0
            if precision < 0.3:  # ignore consequents that follow regardless
                continue
            ctx = {ant}
            pred = {con}
            sig = frozenset(ctx | pred)
            if sig in existing_signatures:
                continue
            hid = f"H{next(self._id_gen)}"
            proposals.append(
                LocalHypothesis(
                    hypothesis_id=hid,
                    context_tokens=set(ctx),
                    action=action,
                    predicted_tokens=set(pred),
                    horizon=max(1, delay),
                )
            )
        return proposals

    def reset_counts(self) -> None:
        self.counts.clear()
        self.antecedent_counts.clear()
        self._buffer.clear()


def specialize(parent: LocalHypothesis, extra_context_token: Token, new_id: str) -> LocalHypothesis:
    """Create a child hypothesis narrowing the parent's context.

    The child inherits a DOCUMENTED subset of parent state: the parent's context
    tokens plus one discriminative extra token, same action, same prediction.
    Evidence starts fresh so the child must re-earn confidence.
    """
    child = LocalHypothesis(
        hypothesis_id=new_id,
        context_tokens=set(parent.context_tokens) | {extra_context_token},
        action=parent.action,
        predicted_tokens=set(parent.predicted_tokens),
        horizon=parent.horizon,
    )
    return child