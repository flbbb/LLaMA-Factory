import math
from typing import Optional

import torch
from torch import nn
from transformers import (
    LogitsProcessorList,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)


class MixtureLogitsProcessor(UnbatchedClassifierFreeGuidanceLogitsProcessor):
    def __init__(
        self,
        mixture_alpha: float,
        mixture_mode: str,
        unconditional_model: Optional[torch.nn.Module] = None,
        unconditional_ids: Optional[torch.LongTensor] = None,
        unconditional_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        n_untouched_logits: Optional[int] = 0,
    ):
        super().__init__(
            model=unconditional_model,
            unconditional_ids=unconditional_ids,
            unconditional_attention_mask=unconditional_attention_mask,
            use_cache=use_cache,
            guidance_scale=mixture_alpha,
        )
        mixture_alpha = float(mixture_alpha)
        self.mixture_alpha = mixture_alpha
        self.mixture_mode = mixture_mode
        self.unconditional_model = unconditional_model
        self.n_untouched_logits = n_untouched_logits
        if mixture_alpha != 0.0 and mixture_alpha != 1.0:
            self.log_alpha = math.log(mixture_alpha)
            self.log_minus_alpha = math.log(1.0 - mixture_alpha)
        else:
            self.log_alpha = 0.0
            self.log_minus_alpha = 0.0

    def __call__(self, input_ids, scores):
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        if self.mixture_alpha == 0.0:
            return scores
        if scores.shape[1] < self.n_untouched_logits:
            return scores

        logits = self.get_unconditional_logits(input_ids)[:, -1, :]

        inf_indices = torch.isinf(scores)
        logits[inf_indices] = float("-Inf")
        unconditional_scores = torch.nn.functional.log_softmax(logits, dim=-1)
        if self.mixture_alpha == 1.0:
            return unconditional_scores

        if self.mixture_mode == "soft":
            scores_processed = torch.logaddexp(
                self.log_minus_alpha + scores, self.log_alpha + unconditional_scores
            )
        elif self.mixture_mode == "hard":
            if torch.rand(1).item() >= self.mixture_alpha:
                scores_processed = scores
            else:
                scores_processed = unconditional_scores
        elif self.mixture_mode == "logprobs":
            scores_processed = (
                1.0 - self.mixture_alpha
            ) * scores + self.mixture_alpha * unconditional_scores
        elif self.mixture_mode == "cad":
            scores_processed = (
                1.0 + self.mixture_alpha
            ) * scores - self.mixture_alpha * unconditional_scores

        return scores_processed


class MixtureDecoder(nn.Module):
    def __init__(
        self,
        model,
        unconditional_model,
        mixture_alpha,
        mixture_mode,
        n_untouched_logits=0,
    ):
        super().__init__()
        self.mixture_alpha = mixture_alpha
        self.mixture_mode = mixture_mode
        self.unconditional_model = unconditional_model
        self.model = model
        self.n_untouched_logits = n_untouched_logits

    def generate(self, **kwargs):
        weak_inputs = kwargs.pop("weak_inputs")

        processor = MixtureLogitsProcessor(
            unconditional_model=self.unconditional_model,
            unconditional_ids=weak_inputs[0]["input_ids"],
            unconditional_attention_mask=weak_inputs[0]["attention_mask"],
            mixture_mode=self.mixture_mode,
            mixture_alpha=self.mixture_alpha,
            n_untouched_logits=self.n_untouched_logits,
        )

        processor = LogitsProcessorList([processor])

        return self.model.generate(logits_processor=processor, **kwargs)

    # method as attribute
    @property
    def device(self):
        return self.model.device

    def compute_transition_scores(self, *args, **kwargs):
        return self.model.compute_transition_scores(*args, **kwargs)
