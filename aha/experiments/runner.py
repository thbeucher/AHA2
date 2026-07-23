"""Generic episode runner coupling an LPCCColony to a hidden-rule environment.

The runner is environment-agnostic: any env exposing reset()/step(action)->
StepResult and an `actions` property works. The colony discovers structure only
from experience; the runner never passes the hidden rule to the colony.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from aha.cells.config import LPCCConfig
from aha.colony.colony import LPCCColony


class Env(Protocol):
    actions: list[str]
    def reset(self) -> dict[str, Any]: ...
    def step(self, action: str) -> Any: ...


@dataclass
class EpisodeResult:
    episode: int
    total_reward: float
    steps: int
    n_living: int
    mean_confidence: float
    structural_events: int


@dataclass
class RunResult:
    episodes: list[EpisodeResult] = field(default_factory=list)
    per_step_reward: list[float] = field(default_factory=list)
    per_step_action: list[str] = field(default_factory=list)
    per_step_correct: list[bool] = field(default_factory=list)

    @property
    def total_reward(self) -> float:
        return sum(e.total_reward for e in self.episodes)


def run_episodes(
    colony: LPCCColony,
    env: Env,
    n_episodes: int,
    correct_action_fn=None,
    on_episode_end=None,
) -> RunResult:
    """Run n_episodes. `correct_action_fn(env)` optionally returns the currently
    correct action for reward-free behavioral scoring."""
    result = RunResult()
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_r = 0.0
        steps = 0
        while not done:
            decision = colony.decide(obs)
            action = decision.chosen_action
            correct = None
            if correct_action_fn is not None:
                correct = correct_action_fn(env)
            step = env.step(action)
            colony.deliver_outcome(step.observation, step.reward, action)
            result.per_step_reward.append(step.reward)
            result.per_step_action.append(action)
            if correct is not None:
                result.per_step_correct.append(action == correct)
            obs = step.observation
            total_r += step.reward
            steps += 1
            done = step.done

        living = colony.living_cells()
        mean_conf = (
            sum(c.hypothesis.confidence for c in living) / len(living) if living else 0.0
        )
        result.episodes.append(
            EpisodeResult(
                episode=ep,
                total_reward=total_r,
                steps=steps,
                n_living=len(living),
                mean_confidence=mean_conf,
                structural_events=colony.structural_event_count,
            )
        )
        if on_episode_end is not None:
            on_episode_end(ep, env, colony)
    return result