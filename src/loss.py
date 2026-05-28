import torch
import torch.nn as nn
import torch.nn.functional as F


class SMECLoss(nn.Module):
    """
    SMEC objective: supervised ranking plus S-XBM similarity preservation.
    """

    def __init__(
        self,
        rank_temperature: float = 0.05,
        preserve_temperature: float = 0.05,
        alpha: float = 0.2,
        top_k: int = 10,
    ):
        super().__init__()
        self.rank_temperature = rank_temperature
        self.preserve_temperature = preserve_temperature
        self.alpha = alpha
        self.top_k = top_k
        self.cross_entropy = nn.CrossEntropyLoss()

    def rank_loss(self, query: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        query = F.normalize(query, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        logits = query @ positive.T / self.rank_temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return self.cross_entropy(logits, labels)

    def similarity_preservation_loss(
        self,
        query_original: torch.Tensor,
        query_compressed: torch.Tensor,
        memory_queue,
        compress_fn,
    ) -> torch.Tensor:
        if memory_queue is None or len(memory_queue) == 0:
            return query_compressed.new_zeros(())

        neighbors_original = memory_queue.retrieve_neighbors(query_original, k=self.top_k)
        if neighbors_original.size(1) == 0:
            return query_compressed.new_zeros(())

        batch_size, neighbor_count, original_dim = neighbors_original.shape
        flat_neighbors = neighbors_original.reshape(batch_size * neighbor_count, original_dim)
        neighbors_compressed = compress_fn(flat_neighbors).reshape(batch_size, neighbor_count, -1)

        teacher_logits = torch.bmm(
            F.normalize(neighbors_original, p=2, dim=-1),
            F.normalize(query_original, p=2, dim=-1).unsqueeze(2),
        ).squeeze(2) / self.preserve_temperature
        student_logits = torch.bmm(
            F.normalize(neighbors_compressed, p=2, dim=-1),
            F.normalize(query_compressed, p=2, dim=-1).unsqueeze(2),
        ).squeeze(2) / self.preserve_temperature

        teacher_distribution = F.softmax(teacher_logits.detach(), dim=1)
        student_log_distribution = F.log_softmax(student_logits, dim=1)
        return F.kl_div(student_log_distribution, teacher_distribution, reduction="batchmean")

    def forward(
        self,
        query_original: torch.Tensor,
        positive_original: torch.Tensor,
        query_compressed: torch.Tensor,
        positive_compressed: torch.Tensor,
        memory_queue=None,
        compress_fn=None,
    ):
        rank = self.rank_loss(query_compressed, positive_compressed)
        preserve = query_compressed.new_zeros(())
        if memory_queue is not None and compress_fn is not None:
            preserve = self.similarity_preservation_loss(
                query_original=query_original,
                query_compressed=query_compressed,
                memory_queue=memory_queue,
                compress_fn=compress_fn,
            )
        return rank + self.alpha * preserve, {"rank_loss": rank.detach(), "preserve_loss": preserve.detach()}


class SMECContrastiveLoss(SMECLoss):
    """Backward-compatible alias for older imports."""

    pass
