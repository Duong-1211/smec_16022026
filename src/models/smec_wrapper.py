import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from .ads import AdaptiveDimensionSelection


class SMECModel(nn.Module):
    """
    Frozen-backbone SMEC wrapper with sequential ADS compression stages.

    Text is first encoded into original full-dimensional embeddings. Compression
    then applies learned ADS stages in order, e.g. 768 -> 384 -> 192. Earlier
    stages can be frozen before training the next transition to match SMRL.
    """

    def __init__(self, model_name: str, embedding_dim: int | None = None, ads_temperature: float = 1.0):
        super().__init__()
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.embedding_dim = embedding_dim or int(self.config.hidden_size)
        self.current_dim = self.embedding_dim
        self.ads_temperature = ads_temperature
        self.compression_stages = nn.ModuleList()
        self.stage_dims: list[tuple[int, int]] = []
        self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def reset_compression(self):
        self.compression_stages = nn.ModuleList()
        self.stage_dims = []
        self.current_dim = self.embedding_dim

    def ensure_stage(self, target_dim: int) -> AdaptiveDimensionSelection:
        if target_dim == self.current_dim:
            if not self.compression_stages:
                raise ValueError("Full dimension has no ADS stage")
            return self.compression_stages[-1]
        if target_dim >= self.current_dim:
            raise ValueError(f"Target dim {target_dim} must be smaller than current dim {self.current_dim}")

        stage = AdaptiveDimensionSelection(self.current_dim, target_dim, temperature=self.ads_temperature)
        self.compression_stages.append(stage)
        self.stage_dims.append((self.current_dim, target_dim))
        self.current_dim = target_dim
        return stage

    def set_ads_target_dim(self, target_dim: int):
        """Backward-compatible helper for loading old-style dimension checkpoints."""
        self.reset_compression()
        if target_dim == self.embedding_dim:
            return
        self.ensure_stage(target_dim)

    def freeze_completed_stages(self):
        for stage in self.compression_stages[:-1]:
            stage.eval()
            for param in stage.parameters():
                param.requires_grad = False
        if self.compression_stages:
            self.compression_stages[-1].train()
            for param in self.compression_stages[-1].parameters():
                param.requires_grad = True

    def trainable_compression_parameters(self):
        if not self.compression_stages:
            return []
        return [p for p in self.compression_stages[-1].parameters() if p.requires_grad]

    def encode_original(self, input_ids, attention_mask) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            embeddings = embeddings / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return F.normalize(embeddings, p=2, dim=-1)

    def compress_embeddings(self, embeddings: torch.Tensor, target_dim: int | None = None) -> torch.Tensor:
        target_dim = target_dim or self.current_dim
        output = F.normalize(embeddings, p=2, dim=-1)
        if target_dim == self.embedding_dim:
            return output

        for stage, (_, out_dim) in zip(self.compression_stages, self.stage_dims):
            output = stage(output)
            if out_dim == target_dim:
                return output

        raise ValueError(f"No compression path available for target dim {target_dim}")

    def forward(self, input_ids, attention_mask, return_original: bool = False):
        original = self.encode_original(input_ids=input_ids, attention_mask=attention_mask)
        compressed = self.compress_embeddings(original)
        if return_original:
            return original, compressed
        return compressed

    def checkpoint_metadata(self) -> dict:
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "current_dim": self.current_dim,
            "stage_dims": self.stage_dims,
            "ads_temperature": self.ads_temperature,
        }

    def load_checkpoint_metadata(self, metadata: dict):
        self.reset_compression()
        self.ads_temperature = metadata.get("ads_temperature", self.ads_temperature)
        for _, target_dim in metadata.get("stage_dims", []):
            self.ensure_stage(int(target_dim))
