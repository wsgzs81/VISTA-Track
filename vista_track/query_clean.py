"""Query-clean temporal state helpers.

Sequence trackers often keep a target query or memory token between frames. That
state is helpful inside one video, but harmful if it leaks across unrelated
random training samples. QueryCleanState makes the reset/update policy explicit.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class QueryCleanState:
    """Manage temporal query state for training and inference.

    Training code should call ``reset()`` at the beginning of each independent
    sampled clip. Inference code can keep the same instance for the whole video.
    """

    state: Optional[torch.Tensor] = None
    detach_on_update: bool = True

    def reset(self) -> None:
        self.state = None

    def get(self) -> Optional[torch.Tensor]:
        return self.state

    def update(self, new_state: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if new_state is None:
            self.state = None
            return None
        self.state = new_state.detach() if self.detach_on_update else new_state
        return self.state


def forward_query_clean(model, template, search_frames, *, token_len=1):
    """Forward a sampled clip while preventing cross-sample query leakage.

    The wrapped model/backbone is expected to accept ``track_query`` and return
    either a tensor output or ``(output, aux_dict)``. This helper intentionally
    resets query state once per call, so callers can use it inside ordinary
    random video samplers without hidden history contamination.
    """

    query = QueryCleanState()
    outputs = []

    for search in search_frames:
        result = model.backbone(
            z=template,
            x=search,
            track_query=query.get(),
            token_len=token_len,
        )
        if isinstance(result, tuple):
            features, aux = result
        else:
            features, aux = result, {}

        feat_last = features[-1] if isinstance(features, list) else features
        if getattr(model.backbone, "add_cls_token", False):
            query.update(feat_last[:, :token_len].clone())

        if hasattr(model, "forward_from_backbone"):
            out = model.forward_from_backbone(features, aux)
        else:
            out = {"features": features, **aux}
        outputs.append(out)

    return outputs
