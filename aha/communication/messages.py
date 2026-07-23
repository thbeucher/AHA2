"""Local communication messages and routing.

Implements assumptions: A6.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal


MessageKind = Literal["belief", "prediction", "urgency", "request"]


@dataclass(frozen=True)
class Message:
    """Inspectably communicated local hypothesis evidence."""

    sender_id: str
    receiver_id: str | None
    belief_id: str
    confidence: float
    prediction_summary: str
    urgency: float
    request: str | None = None
    kind: MessageKind = "belief"


@dataclass
class MessageBus:
    """Sparse message buffer with usefulness accounting."""

    pending: list[Message] = field(default_factory=list)
    usefulness: dict[tuple[str, str], float] = field(default_factory=lambda: defaultdict(float))

    def publish(self, message: Message) -> None:
        self.pending.append(message)

    def collect_for(self, agent_id: str) -> list[Message]:
        mine = [m for m in self.pending if m.receiver_id in (None, agent_id)]
        self.pending = [m for m in self.pending if m.receiver_id not in (None, agent_id)]
        return mine

    def mark_useful(self, sender_id: str, receiver_id: str, amount: float = 0.1) -> None:
        self.usefulness[(sender_id, receiver_id)] += amount
