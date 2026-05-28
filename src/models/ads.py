import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveDimensionSelection(nn.Module):
    """
    Adaptive Dimension Selection (ADS).

    The layer learns a differentiable input-dimension selector for one SMRL
    transition, e.g. 768 -> 384. Training uses Gumbel-Softmax; evaluation uses
    deterministic probabilities so a checkpoint always emits stable embeddings.
    """

    def __init__(self, input_dim: int, output_dim: int, temperature: float = 1.0):
        super().__init__()
        if output_dim > input_dim:
            raise ValueError("output_dim must be <= input_dim")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.temperature = temperature
        self.gate_logits = nn.Parameter(torch.empty(output_dim, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.gate_logits)

    def selection_matrix(self, hard: bool = False) -> torch.Tensor:
        if self.training:
            return F.gumbel_softmax(
                self.gate_logits,
                tau=self.temperature,
                hard=hard,
                dim=-1,
            )

        probabilities = F.softmax(self.gate_logits / self.temperature, dim=-1)
        if not hard:
            return probabilities

        indices = probabilities.argmax(dim=-1)
        return F.one_hot(indices, num_classes=self.input_dim).to(probabilities.dtype)

    def forward(self, x: torch.Tensor, hard: bool = False) -> torch.Tensor:
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {x.size(-1)}")

        selected = x @ self.selection_matrix(hard=hard).T
        return F.normalize(selected, p=2, dim=-1)


class TopKSelector(nn.Module):
    """Select top-k dimensions according to external importance scores."""

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor, importance_scores: torch.Tensor):
        topk_indices = torch.topk(importance_scores, self.k).indices
        return x[:, topk_indices]
