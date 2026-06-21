"""Confidence-gated dynamic template memory."""

from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class DynamicTemplateMemory:
    """Keep the initial template plus a small set of confident updates."""

    max_dynamic: int = 2
    score_threshold: float = 0.55
    update_interval: int = 5
    keep_on_cpu_after: int = 1000
    initial_templates: List[torch.Tensor] = field(default_factory=list)
    dynamic_templates: List[torch.Tensor] = field(default_factory=list)
    dynamic_scores: List[float] = field(default_factory=list)

    def reset(self, initial_template: torch.Tensor) -> None:
        self.initial_templates = [initial_template.detach()]
        self.dynamic_templates = []
        self.dynamic_scores = []

    def select(self, template_number: int) -> List[torch.Tensor]:
        templates = list(self.initial_templates)
        if self.max_dynamic > 0:
            templates.extend(self.dynamic_templates[-self.max_dynamic :])
        return templates[:template_number]

    def maybe_update(self, frame_id: int, template: torch.Tensor, score) -> bool:
        if self.max_dynamic <= 0:
            return False
        if frame_id % max(1, self.update_interval) != 0:
            return False

        score_value = float(score.detach().max().item() if torch.is_tensor(score) else score)
        if score_value < self.score_threshold:
            return False

        stored = template.detach()
        if frame_id > self.keep_on_cpu_after:
            stored = stored.cpu()

        self.dynamic_templates.append(stored)
        self.dynamic_scores.append(score_value)

        if len(self.dynamic_templates) > self.max_dynamic:
            keep = sorted(range(len(self.dynamic_scores)), key=self.dynamic_scores.__getitem__)[-self.max_dynamic :]
            self.dynamic_templates = [self.dynamic_templates[i] for i in keep]
            self.dynamic_scores = [self.dynamic_scores[i] for i in keep]
        return True
