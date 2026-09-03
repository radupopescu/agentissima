"""Each configuration's own recommended sampling defaults (§9 Stage 5B).

The controlled comparison runs greedy (§4.2). When the degenerate-rate
detector fires, Stage 5B re-runs a configuration at *its* recommended
defaults and reports it separately. This module answers what those defaults
are, and it answers it **from the artefact** rather than from a table someone
typed — the same rule the fixtures follow, for the same reason: a hand-copied
constant drifts from the thing it describes and nobody notices.

Two artefact families, two places to look:

- **GGUF** carries `general.sampling.*` keys in its header.
- **MLX** ships a `generation_config.json` beside the weights.

They do not agree, and the two GGUF quantisations of one model do not agree
with each other — Q8_0 recommends `temperature 0.1, top_k 50` and QAD-Q4_0
recommends `temperature 0.2, top_k 80`. This is why "the model's recommended
defaults" is resolved per configuration and never shared between them.

**Anything the artefact does not state keeps its §4.2 controlled value.** An
absent key is not an invitation to guess: neither LFM artefact states `top_p`,
so it stays at 1 for those (Bonsai's GGUF does state it, at 0.85), and `seed`
stays at 1337 so the pass is reproducible. Every
resolved set records which file it came from and which keys that file
actually supplied, so a result can never imply a recommendation the artefact
did not make.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .client import DEFAULT_EXTRA_BODY, DEFAULT_SAMPLING
from .gguf_meta import parse as parse_gguf

MODELS_ROOT = Path.home() / ".lmstudio" / "models"

# GGUF header key -> the request parameter it sets.
_GGUF_KEYS = {
    "general.sampling.temp": "temperature",
    "general.sampling.top_k": "top_k",
    "general.sampling.top_p": "top_p",
    "general.sampling.repeat_penalty": "repeat_penalty",
}

# `generation_config.json` key -> the request parameter it sets. HuggingFace
# spells the penalty `repetition_penalty`; llama.cpp and LM Studio call the
# same knob `repeat_penalty`.
_MLX_KEYS = {
    "temperature": "temperature",
    "top_k": "top_k",
    "top_p": "top_p",
    "repetition_penalty": "repeat_penalty",
}

# Parameters that stay at their controlled value whatever the artefact says.
# `max_tokens` is a budget, not a sampling recommendation, and changing it
# would confound the comparison with a different per-turn ceiling. `seed`
# keeps the pass as reproducible as greedy decoding was.
PINNED = ("max_tokens", "seed")


class SamplingUnavailableError(RuntimeError):
    """The artefact states no sampling defaults, so there is nothing to run."""


@dataclass(frozen=True)
class RecommendedSampling:
    """What one configuration's artefact recommends, and where it came from."""

    config_id: str
    source: str
    """The file the values were read from, relative to the models root."""
    stated: dict[str, float | int] = field(default_factory=dict)
    """Only the parameters the artefact actually states."""

    @property
    def sampling(self) -> dict:
        """The §4.2 controlled set with the stated values applied over it."""
        merged = {**DEFAULT_SAMPLING, **DEFAULT_EXTRA_BODY}
        for name, value in self.stated.items():
            if name not in PINNED:
                merged[name] = value
        return merged

    @property
    def changed_from_controlled(self) -> dict[str, tuple]:
        """Parameter -> (controlled, recommended), for the ones that differ."""
        controlled = {**DEFAULT_SAMPLING, **DEFAULT_EXTRA_BODY}
        return {
            name: (controlled[name], value)
            for name, value in self.sampling.items()
            if controlled.get(name) != value
        }

    @property
    def request_sampling(self) -> dict:
        """The half of `sampling` the OpenAI schema accepts directly."""
        return {k: v for k, v in self.sampling.items() if k in DEFAULT_SAMPLING}

    @property
    def request_extra_body(self) -> dict:
        """The half that is not OpenAI-standard and travels in `extra_body`
        (§4.2): `top_k` and `repeat_penalty`."""
        return {k: v for k, v in self.sampling.items() if k in DEFAULT_EXTRA_BODY}

    def as_record(self) -> dict:
        """The provenance shape written into `environment.json` (§3)."""
        return {
            "source": self.source,
            "stated": dict(self.stated),
            "resolved": self.sampling,
        }


def _from_gguf(path: Path) -> dict:
    """GGUF stores these as float32, so `0.1` reads back as
    `0.10000000149011612`. Rounded to 6 decimals: well inside float32's
    precision, and it keeps the recorded value legible as the recommendation
    it is."""
    _version, _n_tensors, metadata, _names = parse_gguf(path)
    stated = {}
    for key, param in _GGUF_KEYS.items():
        value = metadata.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        stated[param] = round(value, 6) if isinstance(value, float) else value
    return stated


def _from_generation_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    return {
        param: config[key]
        for key, param in _MLX_KEYS.items()
        if isinstance(config.get(key), (int, float))
    }


def resolve(
    config_id: str, model_path: str, *, models_root: Path | None = None
) -> RecommendedSampling:
    """Read `model_path`'s own recommended sampling defaults.

    `model_path` is §2.1's stable identifier — the path under the models root,
    never LM Studio's `modelKey`.
    """
    root = Path(models_root or MODELS_ROOT)
    artefact = root / model_path

    if artefact.suffix == ".gguf":
        if not artefact.is_file():
            raise SamplingUnavailableError(f"{artefact} does not exist")
        stated = _from_gguf(artefact)
        source = model_path
    else:
        generation_config = artefact / "generation_config.json"
        if not generation_config.is_file():
            raise SamplingUnavailableError(
                f"{generation_config} does not exist; this artefact states no "
                "sampling defaults"
            )
        stated = _from_generation_config(generation_config)
        source = f"{model_path}/generation_config.json"

    if not stated:
        raise SamplingUnavailableError(
            f"{source} states no sampling defaults — there is nothing for a "
            "Stage 5B sampling pass to run that differs from §4.2"
        )
    return RecommendedSampling(config_id=config_id, source=source, stated=stated)
