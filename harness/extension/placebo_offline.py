"""Offline backends for Addendum G's segments (no API calls).

`StandInTruth` (roster params `standin_truth = true`, for kinds "control" and "simulated") gives the
offline backends a truth where a dataset has none. wdc-B's predicates have no machine-checkable rule, so
semaudit's control arm and simulator cannot answer its filters. With the wrapper, a payload whose p or q
has no rule gets a stand-in: a hidden field per record, set from a hash of the record id (True for about
half the records, independently for p and q), and rules that read it. Rules the dataset has are kept. The
stand-in exists only to exercise the pipeline offline; it says nothing about the data.

`PlaceboSimulator` is semaudit's SimulatedBackend with the extra parameters below (roster kind
"simulated" with any of them). With every extra parameter at its default it answers exactly as semaudit's
simulator: the same requests get the same answers. Reworded requests carry the payload of the claim they
restate, so under the null they are drawn from that claim's distribution, and every comparison of
Addendum G compares exchangeable draws.

| Parameter | Effect |
|---|---|
| paraphrase_effect | P(True) of p_para1, p_para2 and pq_para shifted by this amount, with a random sign per row and condition |
| template_effect | the same for p_tmpl |
| order_effect | the same for p_order |
| double_rejection | share of rows the model rejects whatever the polarity: False to p, not p and the conjunctions (and their rewordings) in the two-valued styles; in the `unknown` style, Unknown with probability `unknown_absorbs_rejection` per answer, else False |
| unknown_rate, unknown_rate_easy | in the `unknown` style, the probability that an answer about p or q is Unknown, on the rows the simulator finds hard (semaudit's hard_frac) and on the others |
| noise_salt | a different noise stream with the same item-level distributions (a second arm under the null) |

In the `unknown` style a claim about p or q is answered from one three-valued judgment (rejection state,
then Unknown, then True with semaudit's probability); not p negates it (semaudit's negation_bias still
lowers True); a conjunction combines two independent judgments with Kleene's AND, and semaudit's
fusion_bias then turns True into False with the probability that lowers P(True) by that amount. Under the
null, F3(not p) is therefore distributed as NOT3 of F3(p), and F3(p and q) as AND3 of independent F3(p)
and F3(q).
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

from semaudit.core import Completion, Task, stable_seed
from semaudit.offline import SimulatedBackend, _clip, _render

from .placebo import and3
from .styles import UNKNOWN, UNKNOWN_ANSWER

STANDIN_FIELDS = {"p": "__standin_p", "q": "__standin_q"}


def standin_value(record_id: str, claim: str) -> bool:
    """The stand-in truth of p or q for a record: a fixed function of the record id."""
    return random.Random(stable_seed("standin-truth", claim, record_id)).random() < 0.5


def _claim_rule(claim: str | None, p: dict[str, Any], q: dict[str, Any]) -> dict[str, Any] | None:
    return {"p": p, "not_p": {"not": p}, "q": q, "p_and_q": {"and": [p, q]},
            "q_and_p": {"and": [p, q]}}.get(claim or "")


def standin_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """The payload with stand-in rules for p and q where the dataset has none (unchanged otherwise)."""
    rules = payload.get("rules")
    if not isinstance(rules, dict) or (rules.get("p") and rules.get("q")):
        return payload
    full = {c: rules.get(c) or {"field": STANDIN_FIELDS[c], "equals": "1"} for c in ("p", "q")}

    def with_fields(record: dict[str, str], rid: str) -> dict[str, str]:
        return {**record, **{STANDIN_FIELDS[c]: "1" if standin_value(rid, c) else "0" for c in ("p", "q")}}

    new = {**payload, "rules": full}
    if "record" in payload:
        new["record"] = with_fields(payload["record"], payload["record_id"])
        rule = _claim_rule(payload.get("claim"), full["p"], full["q"])
        if rule is not None:
            new["rule"] = rule
    if "records" in payload:  # batch and COUNT payloads judge p
        new["records"] = [with_fields(r, rid) for r, rid in zip(payload["records"], payload["record_ids"])]
        new["rule"] = full["p"]
    return new


class StandInTruth:
    """Wraps an offline backend; payloads without rules for p or q get the stand-in truth."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.name = inner.name
        self.concurrency = getattr(inner, "concurrency", None)

    def describe(self) -> dict[str, Any]:
        desc = self.inner.describe()
        return {**desc, "params": {**desc["params"], "standin_truth": True}}

    def bind_run(self, run_id: str) -> None:
        if hasattr(self.inner, "bind_run"):
            self.inner.bind_run(run_id)

    def complete(self, task: Task) -> Completion:
        return self.inner.complete(replace(task, payload=standin_payload(task.payload)))


SIM_PARAMS = ("paraphrase_effect", "template_effect", "order_effect", "double_rejection", "unknown_rate",
              "unknown_rate_easy", "unknown_absorbs_rejection", "noise_salt")
SHIFT_PARAM = {"p_para1": "paraphrase_effect", "p_para2": "paraphrase_effect", "pq_para": "paraphrase_effect",
               "p_tmpl": "template_effect", "p_order": "order_effect"}
P_CLAIMS = ("p", "not_p", "p_and_q", "q_and_p")  # claims about p: the ones a rejected row is False to


class PlaceboSimulator(SimulatedBackend):
    def __init__(self, name: str = "sim", *, paraphrase_effect: float = 0.0, template_effect: float = 0.0,
                 order_effect: float = 0.0, double_rejection: float = 0.0, unknown_rate: float = 0.0,
                 unknown_rate_easy: float = 0.0, unknown_absorbs_rejection: float = 0.0, noise_salt: str = "",
                 **params: Any) -> None:
        super().__init__(name, **params)
        self.extra = {"paraphrase_effect": float(paraphrase_effect), "template_effect": float(template_effect),
                      "order_effect": float(order_effect), "double_rejection": float(double_rejection),
                      "unknown_rate": float(unknown_rate), "unknown_rate_easy": float(unknown_rate_easy),
                      "unknown_absorbs_rejection": float(unknown_absorbs_rejection), "noise_salt": str(noise_salt)}
        self.salt = self._salted("")

    def _salted(self, run_id: str) -> str:
        salt = self.extra["noise_salt"]
        return f"{run_id}\x1f{salt}" if salt else run_id

    def bind_run(self, run_id: str) -> None:
        self.salt = self._salted(run_id)

    def describe(self) -> dict[str, Any]:
        desc = super().describe()
        return {**desc, "params": {**desc["params"], **self.extra}}

    # --- item-level properties --------------------------------------------------------------------

    def _hard(self, key: str) -> bool:  # semaudit's choice of ambiguous items (as in _flip_rate)
        return random.Random(stable_seed(self.params.seed, "hard", key)).random() < self.params.hard_frac

    def _rejected(self, rid: str) -> bool:
        return random.Random(stable_seed(self.params.seed, "reject", rid)).random() < self.extra["double_rejection"]

    def _changes(self, task: Task) -> bool:
        if task.context.get("style") == UNKNOWN or self.extra["double_rejection"] > 0:
            return True
        shift = SHIFT_PARAM.get(task.condition)
        return bool(shift and self.extra[shift])

    # --- two-valued styles ------------------------------------------------------------------------

    def _p2(self, p: dict[str, Any], condition: str) -> float:
        prob = self._p_filter(p)  # semaudit's probability, with its negation and fusion biases
        if p["claim"] in P_CLAIMS and self._rejected(p["record_id"]):
            prob = 0.0
        shift = SHIFT_PARAM.get(condition)
        if shift:
            prob += self.extra[shift] * self._sign(f"{p['record_id']}:{condition}")
        return prob

    # --- the unknown style ------------------------------------------------------------------------

    def _u_rate(self, key: str) -> float:
        return self.extra["unknown_rate"] if self._hard(key) else self.extra["unknown_rate_easy"]

    def _judge(self, claim: str, p: dict[str, Any], rng: random.Random) -> Any:
        """One three-valued answer to p, not p or q."""
        rid, target = p["record_id"], ("q" if claim == "q" else "p")
        key = f"{rid}:{target}"
        if target == "p" and self._rejected(rid):
            return UNKNOWN_ANSWER if rng.random() < self.extra["unknown_absorbs_rejection"] else False
        if rng.random() < self._u_rate(key):
            return UNKNOWN_ANSWER
        prob = self._p_claim(p["rules"][target], p["record"], key)
        if claim == "not_p":
            return rng.random() < _clip(1 - prob - self.params.negation_bias)
        return rng.random() < prob

    def _p_true(self, target: str, p: dict[str, Any]) -> float:
        rid = p["record_id"]
        if target == "p" and self._rejected(rid):
            return 0.0
        key = f"{rid}:{target}"
        return (1 - self._u_rate(key)) * self._p_claim(p["rules"][target], p["record"], key)

    def _draw3(self, p: dict[str, Any], rng: random.Random) -> Any:
        claim = p["claim"]
        if claim in ("p", "not_p", "q"):
            return self._judge(claim, p, rng)
        value = and3(self._judge("p", p, rng), self._judge("q", p, rng))
        fb = self.params.fusion_bias
        if value is True and fb > 0:
            t = self._p_true("p", p) * self._p_true("q", p)
            if t > 0 and rng.random() < min(1.0, fb / t):
                value = False
        return value

    def complete(self, task: Task) -> Completion:
        if task.op != "filter" or not self._changes(task):
            return super().complete(task)
        rng = random.Random(stable_seed(self.params.seed, self.salt, task.task_id))
        if task.context.get("style") == UNKNOWN:
            value = self._draw3(task.payload, rng)
            text = "Unknown" if value == UNKNOWN_ANSWER else ("True" if value else "False")
            return Completion(text=text, served_model=self.model)
        value = rng.random() < _clip(self._p2(task.payload, task.condition))
        return Completion(text=_render(task, [value]), served_model=self.model)
