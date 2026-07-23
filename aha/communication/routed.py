"""Routed, costly, delayed, lossy, bandwidth-limited communication.

This is NOT a free global broadcast. It is the mechanism under test for
interaction-dependent emergence.

A CausalMessage carries a proposition (a token) that a receiving LPCC can fold
into its LOCAL temporal context. In the distributed-chain task, a view-B cell
whose stage just became live can emit a proposition token like
("msg:B", "live"); a view-C cell can then key a hypothesis on that received
token as CONTEXT, allowing the chain to complete without any cell seeing the
whole state.

Ablation hooks (all in MessageRouter): disable, randomize, delay, restrict
bandwidth, corrupt confidence/proposition. Every delivered message is logged for
information-flow analysis and message-level causal ablation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from aha.cells.patterns import Token


@dataclass
class CausalMessage:
    sender_id: str
    hypothesis_id: str
    proposition: Token          # a token the receiver can add to its context
    confidence: float
    prediction: Token | None    # what the sender expects to follow
    timestamp: int
    validity_horizon: int       # message is usable for this many steps
    uncertainty: float

    def expired(self, now: int) -> bool:
        return now > self.timestamp + self.validity_horizon


@dataclass
class DeliveryRecord:
    """Full record of one delivered message for causal analysis."""

    message: CausalMessage
    receiver_id: str
    deliver_time: int
    receiver_state_before: dict[str, Any]
    receiver_state_after: dict[str, Any] = field(default_factory=dict)
    tendency_before: float | None = None
    tendency_after: float | None = None
    later_reward: float = 0.0


@dataclass
class RouterConfig:
    enabled: bool = True
    randomize: bool = False       # replace propositions with random tokens
    delay: int = 1                # steps between send and delivery
    loss_prob: float = 0.0        # probability a message is dropped
    bandwidth: int = 8            # max messages delivered per receiver per step
    corrupt_confidence: bool = False
    corrupt_proposition: bool = False
    energy_cost: float = 0.02     # charged to sender per message


class MessageRouter:
    """Delivers messages with delay/loss/bandwidth and records everything."""

    def __init__(self, config: RouterConfig | None = None, seed: int = 0):
        self.config = config or RouterConfig()
        self.rng = random.Random(seed)
        # queue: deliver_time -> list of (message, intended_receiver_or_None)
        self._queue: dict[int, list[tuple[CausalMessage, str | None]]] = {}
        self.delivery_log: list[DeliveryRecord] = []
        self._random_token_pool = [
            ("msg:noise", str(i)) for i in range(16)
        ]
        self.sent_count = 0
        self.delivered_count = 0

    def send(self, message: CausalMessage, receiver_id: str | None, now: int) -> float:
        """Queue a message. Returns energy cost charged to the sender."""
        if not self.config.enabled:
            return 0.0
        if self.rng.random() < self.config.loss_prob:
            return self.config.energy_cost  # cost paid, message lost
        msg = message
        if self.config.randomize:
            msg = self._randomized(message, now)
        elif self.config.corrupt_confidence:
            msg = self._corrupt_conf(message)
        elif self.config.corrupt_proposition:
            msg = self._corrupt_prop(message, now)
        dt = now + self.config.delay
        self._queue.setdefault(dt, []).append((msg, receiver_id))
        self.sent_count += 1
        return self.config.energy_cost

    def deliver(self, now: int, receiver_ids: list[str]) -> dict[str, list[CausalMessage]]:
        """Return, per receiver, the messages delivered at `now` (bandwidth-capped)."""
        pending = self._queue.pop(now, [])
        by_receiver: dict[str, list[CausalMessage]] = {rid: [] for rid in receiver_ids}
        # Broadcast (receiver None) goes to all; else to the named receiver.
        for msg, rid in pending:
            targets = receiver_ids if rid is None else [rid]
            for t in targets:
                if t in by_receiver and len(by_receiver[t]) < self.config.bandwidth:
                    by_receiver[t].append(msg)
                    self.delivered_count += 1
        return by_receiver

    def log_delivery(self, record: DeliveryRecord) -> None:
        self.delivery_log.append(record)

    # -- corruption helpers ------------------------------------------------- #
    def _randomized(self, m: CausalMessage, now: int) -> CausalMessage:
        tok = self.rng.choice(self._random_token_pool)
        return CausalMessage(
            sender_id=m.sender_id, hypothesis_id="RANDOM", proposition=tok,
            confidence=self.rng.random(), prediction=None, timestamp=now,
            validity_horizon=m.validity_horizon, uncertainty=1.0,
        )

    def _corrupt_conf(self, m: CausalMessage) -> CausalMessage:
        return CausalMessage(
            sender_id=m.sender_id, hypothesis_id=m.hypothesis_id, proposition=m.proposition,
            confidence=self.rng.random(), prediction=m.prediction, timestamp=m.timestamp,
            validity_horizon=m.validity_horizon, uncertainty=m.uncertainty,
        )

    def _corrupt_prop(self, m: CausalMessage, now: int) -> CausalMessage:
        tok = self.rng.choice(self._random_token_pool)
        return CausalMessage(
            sender_id=m.sender_id, hypothesis_id=m.hypothesis_id, proposition=tok,
            confidence=m.confidence, prediction=m.prediction, timestamp=m.timestamp,
            validity_horizon=m.validity_horizon, uncertainty=m.uncertainty,
        )