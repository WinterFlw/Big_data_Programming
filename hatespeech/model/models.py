from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from src.utils import NUM_LABELS, VADER_COLUMNS


def _force_eager_attention(model: nn.Module) -> None:
    """Attention supervision/XAI가 attention weights를 받을 수 있게 SDPA 대신 eager attention을 씁니다."""
    encoder = getattr(model, "encoder", None)
    if encoder is None or not hasattr(encoder, "config"):
        return
    encoder.config._attn_implementation = "eager"
    layer_container = getattr(getattr(encoder, "encoder", None), "layer", [])
    for layer in layer_container:
        attention = getattr(getattr(layer, "attention", None), "self", None)
        if attention is not None:
            attention._attn_implementation = "eager"



class TransformerCLSClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int = NUM_LABELS,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(model_name)
        _force_eager_attention(self)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        hidden_size = self.encoder.config.hidden_size  # BERT-base는 768
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)  # 768 -> 3

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = getattr(outputs, "pooler_output", None)
        if pooled_output is None:
            pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)  # logits 반환 (softmax 전!)


class HybridSentimentClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int = NUM_LABELS,
        dropout: float = 0.1,
        hidden_dim: int = 256,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(model_name)
        _force_eager_attention(self)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        hidden_size = self.encoder.config.hidden_size  # 768 for BERT-base
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(hidden_size + len(VADER_COLUMNS), hidden_dim)
        self.relu = nn.ReLU()  # 비선형 활성화 — MLP의 표현력을 높여줘요
        self.out = nn.Linear(hidden_dim, num_labels)  # 256 -> 3 (최종 분류)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, vader: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = getattr(outputs, "pooler_output", None)
        if pooled_output is None:
            pooled_output = outputs.last_hidden_state[:, 0, :]
        combined = torch.cat([pooled_output, vader], dim=1)  # [batch, 772]
        combined = self.dropout(combined)
        combined = self.hidden(combined)   # [batch, 256]
        combined = self.relu(combined)     # 비선형 변환
        return self.out(combined)          # [batch, 3] logits


# ╔══════════════════════════════════════════════════════════╗
# ║  Ablation 모델 — MLP 용량 효과를 분리해요!               ║
# ╚══════════════════════════════════════════════════════════╝
# BERT+VADER가 좋아진 게 VADER 4차원 덕분인지, 아니면 단순히 MLP가 커서인지
# 알아보기 위한 대조 실험 모델이에요!
#
# 구조: [CLS](768d) → Dropout → Linear(768, 256) → ReLU → Linear(256, 3)
#
# HybridSentimentClassifier와 동일한 MLP 구조이지만 VADER 입력이 없어요.
# 이 모델이 BERT+VADER와 비슷한 성능을 낸다면 → MLP 크기 효과
# 이 모델보다 BERT+VADER가 확실히 낫다면 → VADER의 실질적 기여 입증!

class TransformerMLPClassifier(nn.Module):
    """
    Ablation 분류기: Transformer [CLS] → MLP → 3-class (VADER 없이).

    BERT+VADER와 동일한 MLP 용량을 가지되 VADER 입력을 제거하여,
    성능 향상이 MLP 크기 때문인지 VADER 피처 때문인지 분리합니다.

    구조:
      [CLS](768d) → Dropout → Linear(768, hidden_dim) → ReLU → Linear(hidden_dim, 3)
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int = NUM_LABELS,
        dropout: float = 0.1,
        hidden_dim: int = 256,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(model_name)
        _force_eager_attention(self)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        # VADER 없이 768d에서 바로 MLP로! (Hybrid는 772d → 256)
        self.hidden = nn.Linear(hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = getattr(outputs, "pooler_output", None)
        if pooled_output is None:
            pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.hidden(pooled_output)
        pooled_output = self.relu(pooled_output)
        return self.out(pooled_output)


class TransformerConditionClassifier(nn.Module):
    """v2.1 8조건 공통 모델: MLP head를 고정하고 VADER/Aux head만 조건별로 켭니다."""

    def __init__(
        self,
        model_name: str,
        use_vader: bool = False,
        num_labels: int = NUM_LABELS,
        dropout: float = 0.1,
        hidden_dim: int = 256,
        freeze_encoder: bool = False,
        num_targets: int = 0,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.use_vader = use_vader
        self.num_targets = num_targets
        self.encoder = AutoModel.from_pretrained(model_name)
        _force_eager_attention(self)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        hidden_size = self.encoder.config.hidden_size
        input_dim = hidden_size + (len(VADER_COLUMNS) if use_vader else 0)
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, num_labels)
        self.target_head = nn.Linear(hidden_dim, num_targets) if num_targets > 0 else None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vader: torch.Tensor | None = None,
        output_attentions: bool = False,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | tuple[torch.Tensor, ...] | None]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True,
        )
        pooled_output = getattr(outputs, "pooler_output", None)
        if pooled_output is None:
            pooled_output = outputs.last_hidden_state[:, 0, :]
        if self.use_vader:
            if vader is None:
                raise ValueError("vader tensor is required for VADER-enabled conditions")
            pooled_output = torch.cat([pooled_output, vader], dim=1)
        hidden = self.relu(self.hidden(self.dropout(pooled_output)))
        logits = self.out(hidden)
        if not return_dict:
            return logits
        return {
            "logits": logits,
            "target_logits": self.target_head(hidden) if self.target_head is not None else None,
            "attentions": outputs.attentions if output_attentions else None,
        }


@dataclass(frozen=True)
class ConditionSpec:
    condition: str
    family: str
    model_name: str
    use_attention_loss: bool
    use_vader: bool
    use_target_aux: bool = False


V2_CONDITION_SPECS: list[ConditionSpec] = [
    ConditionSpec("A_B", "BERT", "bert-base-uncased", False, False),
    ConditionSpec("B_B", "BERT", "bert-base-uncased", True, False),
    ConditionSpec("C_B", "BERT", "bert-base-uncased", False, True),
    ConditionSpec("D_B", "BERT", "bert-base-uncased", True, True),
    ConditionSpec("A_R", "RoBERTa", "roberta-base", False, False),
    ConditionSpec("B_R", "RoBERTa", "roberta-base", True, False),
    ConditionSpec("C_R", "RoBERTa", "roberta-base", False, True),
    ConditionSpec("D_R", "RoBERTa", "roberta-base", True, True),
]
