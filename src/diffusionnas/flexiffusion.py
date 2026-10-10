"""Flexiffusion segment-wise execution schedules for SD1.5 (training-free adaptation).

Reimplements the schedule semantics of Flexiffusion (Huang, Chang and Yao,
arXiv:2409.17566, Sec. 3-4.1 and Appendix E.1); no official code was released.
A schedule is ``N-segment`` equal-length segments. Each segment starts with a
full U-Net step, continues with partial steps that reuse DeepCache features
cached at ``branch``, and ends with null steps that call no network at all.
``interval`` counts the active (full + partial) steps in a segment.

Adaptation choices (not specified by the paper):
- Sampler is deterministic DDIM (eta=0), not the paper's 50-step PLMS. A null
  step returns the previous noise prediction; for deterministic DDIM this is
  exactly equivalent to skipping that timestep (the predicted x0 is unchanged),
  so it matches the paper's "skips all the computation".
- ``branch`` uses DeepCache's ``cache_branch_id`` numbering (0-11 for SD1.5).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FULL = "full"
PARTIAL = "partial"
NULL = "null"

# SD1.5 search space from Flexiffusion Table 8 (Parti-Prompts and MS-COCO).
SD15_SEGMENT_LENGTH = 3
SD15_N_SEGMENT_BY_BUDGET = {"90G": (9, 10, 11), "80G": (8, 9, 10), "70G": (7, 8, 9)}
SD15_BRANCH_CHOICES = (1, 3, 6)
SD15_INTERVAL_CHOICES = (2, 3)


@dataclass(frozen=True)
class StepAction:
    kind: str
    branch: int | None = None


def uniform_segments(n_segment: int, branch: int, interval: int) -> list[dict[str, int]]:
    """Segments with one shared setting, i.e. the uniform slice of the search space."""
    return [{"branch": branch, "interval": interval} for _ in range(n_segment)]


def expand_schedule(segments: list[dict[str, int]], segment_length: int) -> list[StepAction]:
    """Expand segment settings into one action per denoising step."""
    actions: list[StepAction] = []
    for segment in segments:
        interval, branch = segment["interval"], segment["branch"]
        if not 1 <= interval <= segment_length:
            raise ValueError(f"interval must be in [1, {segment_length}], got {interval}")
        actions.append(StepAction(FULL))
        actions.extend(StepAction(PARTIAL, branch) for _ in range(interval - 1))
        actions.extend(StepAction(NULL) for _ in range(segment_length - interval))
    return actions


def schedule_stats(actions: list[StepAction]) -> dict[str, int]:
    counts = {kind: sum(action.kind == kind for action in actions) for kind in (FULL, PARTIAL, NULL)}
    return {
        "steps": len(actions),
        "full_steps": counts[FULL],
        "partial_steps": counts[PARTIAL],
        "null_steps": counts[NULL],
        "nfe": counts[FULL] + counts[PARTIAL],
    }


def config_actions(config: dict[str, Any]) -> list[StepAction]:
    flex = config["flexiffusion"]
    return expand_schedule(flex["segments"], flex["segment_length"])


def build_helper(pipe: Any, actions: list[StepAction]) -> Any:
    """Return an enabled-ready DeepCache helper driven by a per-step schedule."""
    from DeepCache import DeepCacheSDHelper

    class FlexiffusionSDHelper(DeepCacheSDHelper):
        def is_skip_step(self, block_i, layer_i, blocktype="down"):
            action = actions[self.cur_timestep]
            if action.kind == FULL:
                return False
            # Partial step: DeepCache's skip rule for the segment's branch.
            cache_block_id, cache_layer_id = divmod(action.branch, 3)
            if block_i > cache_block_id or blocktype == "mid":
                return True
            if block_i < cache_block_id:
                return False
            return layer_i >= cache_layer_id if blocktype == "down" else layer_i > cache_layer_id

        def wrap_unet_forward(self):
            self.function_dict["unet_forward"] = self.pipe.unet.forward

            def wrapped_forward(*args, **kwargs):
                timesteps = list(self.pipe.scheduler.timesteps)
                if len(timesteps) != len(actions):
                    raise RuntimeError(f"schedule has {len(actions)} actions for {len(timesteps)} timesteps")
                self.cur_timestep = timesteps.index(args[1].item())
                if actions[self.cur_timestep].kind == NULL:
                    return self.last_unet_output
                self.last_unet_output = self.function_dict["unet_forward"](*args, **kwargs)
                return self.last_unet_output

            self.pipe.unet.forward = wrapped_forward

        def reset_states(self):
            super().reset_states()
            self.last_unet_output = None

    return FlexiffusionSDHelper(pipe=pipe)
