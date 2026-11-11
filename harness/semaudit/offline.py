"""Offline backends: a deterministic control arm and a stochastic simulator.

The control arm implements every operator as compiled code, so every identity holds exactly; a run
against it must report zero violations, which checks the harness end to end. The simulator draws
answers from per-item probabilities with optional injected effects, which checks that the
statistics stay quiet under pure noise and detect effects of known size.
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass
from typing import Any

from .core import Completion, Task, stable_seed
from .data import eval_rule
from .prompts import STYLES

_TOKEN = re.compile(r"[a-z0-9]+")


def canonical(value: str) -> str:
    """Idempotent normalizer used as the compiled map operator."""
    return " ".join(_TOKEN.findall(value.lower()))


def same_entity(left: dict[str, str], right: dict[str, str]) -> bool:
    if left.get("cluster") and right.get("cluster"):
        return left["cluster"] == right["cluster"]
    return canonical(left.get("title", "")) == canonical(right.get("title", ""))


def _render(task: Task, values: list[Any]) -> str:
    """Answer in the words of the task's prompt style, as a model following it would."""
    style = STYLES[task.context.get("style", "lotus")]
    yes, no = style.yes.capitalize(), style.no.capitalize()
    if task.op in ("filter", "match"):
        return yes if values[0] else no
    if task.op == "filter_batch":
        return "\n".join(f"{i}: {yes if v else no}" for i, v in enumerate(values, 1))
    return str(values[0])  # count or map


class ControlBackend:
    """Compiled operators: deterministic and logically consistent."""

    provider = "offline"

    def __init__(self, name: str = "control") -> None:
        self.name = name
        self.model = "control"

    def describe(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "params": {}}

    def complete(self, task: Task) -> Completion:
        p = task.payload
        if task.op == "filter":
            values = [eval_rule(p["rule"], p["record"])]
        elif task.op == "match":
            values = [same_entity(p["left"], p["right"])]
        elif task.op == "filter_batch":
            values = [eval_rule(p["rule"], r) for r in p["records"]]
        elif task.op == "count":
            values = [sum(eval_rule(p["rule"], r) for r in p["records"])]
        elif task.op == "map":
            values = [canonical(p["value"])]
        else:
            raise ValueError(f"unknown op: {task.op}")
        return Completion(text=_render(task, values), served_model=self.model)


@dataclass
class SimParams:
    seed: int = 0
    hard_frac: float = 0.15  # share of items the simulated model finds ambiguous
    hard_flip: float = 0.35  # per-call error rate on ambiguous items
    easy_flip: float = 0.01  # per-call error rate on the rest
    order_bias: float = 0.0  # match: P(True) shift when Record 1 has the longer title
    neighbour_effect: float = 0.0  # batch: P(True) shift toward the share of true neighbours
    position_effect: float = 0.0  # batch: P(True) shift from first to last position
    negation_bias: float = 0.0  # negated claims: P(True) lowered
    fusion_bias: float = 0.0  # conjunctive claims: P(True) lowered
    context_effect: float = 0.0  # full joined-row context: per-item shift of random sign
    map_drift: float = 0.0  # share of values whose rewrite drops another token when re-applied
    map_noise: float = 0.0  # per-call chance of an equivalent formatting variant (idempotent per output)
    call_failure: float = 0.0  # batch: per-call chance that every row in the call gets the same label
    count_bias: float = 0.0  # aggregate counts: added error per record in the batch


def _clip(x: float) -> float:
    return min(1.0, max(0.0, x))


class SimulatedBackend:
    """Draws each answer from a per-item probability; effects are injected via SimParams.

    With every effect at zero the paired conditions of each identity have identical output
    distributions (exchangeability), which is exactly the null hypothesis the analysis tests.
    """

    provider = "offline"

    def __init__(self, name: str = "sim", **params: Any) -> None:
        self.name = name
        self.model = "sim"
        self.params = SimParams(**params)
        self.salt = ""

    def bind_run(self, run_id: str) -> None:
        """Independent runs must draw independent noise."""
        self.salt = run_id

    def describe(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "params": asdict(self.params)}

    def _flip_rate(self, key: str) -> float:
        hard = random.Random(stable_seed(self.params.seed, "hard", key)).random() < self.params.hard_frac
        return self.params.hard_flip if hard else self.params.easy_flip

    def _p_claim(self, rule: dict[str, Any], record: dict[str, str], key: str) -> float:
        flip = self._flip_rate(key)
        return 1 - flip if eval_rule(rule, record) else flip

    def _sign(self, key: str) -> int:
        return 1 if random.Random(stable_seed(self.params.seed, "sign", key)).random() < 0.5 else -1

    def _p_filter(self, p: dict[str, Any]) -> float:
        rid, rec, rules = p["record_id"], p["record"], p["rules"]
        pp = self._p_claim(rules["p"], rec, f"{rid}:p")
        claim = p["claim"]
        if claim == "p":
            prob = pp
        elif claim == "not_p":
            prob = 1 - pp - self.params.negation_bias
        elif claim == "q":
            prob = self._p_claim(rules["q"], rec, f"{rid}:q")
        else:  # a conjunction: under the null, as if each conjunct were judged independently
            prob = pp * self._p_claim(rules["q"], rec, f"{rid}:q") - self.params.fusion_bias
        if p.get("serialization") == "joined_full":
            prob += self.params.context_effect * self._sign(rid)
        return prob

    def _p_match(self, p: dict[str, Any]) -> float:
        left, right = p["left"], p["right"]
        flip = self._flip_rate("|".join(sorted([p["left_id"], p["right_id"]])))
        prob = 1 - flip if same_entity(left, right) else flip
        if len(left.get("title", "")) > len(right.get("title", "")):
            prob += self.params.order_bias
        return prob

    def _p_batch(self, p: dict[str, Any]) -> list[float]:
        recs, ids, rule = p["records"], p["record_ids"], p["rules"]["p"]
        truths = [eval_rule(rule, r) for r in recs]
        n = len(recs)
        probs = []
        for i, (rec, rid) in enumerate(zip(recs, ids)):
            prob = self._p_claim(rule, rec, f"{rid}:p")
            others = truths[:i] + truths[i + 1:]
            if others:
                prob += self.params.neighbour_effect * (sum(others) / len(others) - 0.5)
            if n > 1:
                prob += self.params.position_effect * (i / (n - 1) - 0.5)
            probs.append(prob)
        return probs

    def _map(self, value: str, rng: random.Random) -> str:
        """Canonicalize; with `map_noise`, sometimes return the upper-case variant instead. An output
        that is already canonical comes back unchanged, so the map is idempotent per output unless
        `map_drift` makes re-application drop a token."""
        c = canonical(value)
        if value in (c, c.upper()):
            out = value
        else:
            out = c.upper() if rng.random() < self.params.map_noise else c
        drifts = random.Random(stable_seed(self.params.seed, "drift", canonical(out))).random() < self.params.map_drift
        if drifts and len(out.split()) > 1:
            out = " ".join(out.split()[:-1])
        return out

    def complete(self, task: Task) -> Completion:
        rng = random.Random(stable_seed(self.params.seed, self.salt, task.task_id))
        p = task.payload
        if task.op == "filter":
            values: list[Any] = [rng.random() < _clip(self._p_filter(p))]
        elif task.op == "match":
            values = [rng.random() < _clip(self._p_match(p))]
        elif task.op == "filter_batch":
            if rng.random() < self.params.call_failure:  # the whole call collapses to one label
                label = rng.random() < 0.5
                values = [label] * len(p["records"])
            else:
                values = [rng.random() < _clip(x) for x in self._p_batch(p)]
        elif task.op == "count":
            rule = p["rules"]["p"]
            hits = sum(rng.random() < self._p_claim(rule, r, f"{rid}:p")
                       for r, rid in zip(p["records"], p["record_ids"]))
            values = [hits + round(self.params.count_bias * len(p["records"]))]
        elif task.op == "map":
            values = [self._map(p["value"], rng)]
        else:
            raise ValueError(f"unknown op: {task.op}")
        return Completion(text=_render(task, values), served_model=self.model)
