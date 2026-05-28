import torch
import torch.nn.functional as F


class SelectiveCrossBatchMemory:
    """
    Selective Cross-Batch Memory (S-XBM).

    The queue stores original high-dimensional embeddings from the frozen
    backbone. Retrieval uses original-space similarity and never returns
    unfilled/random entries.
    """

    def __init__(self, memory_size: int, embedding_dim: int, device: torch.device):
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        self.device = device
        self.memory_queue = torch.empty(memory_size, embedding_dim, device=device)
        self.ptr = 0
        self.count = 0

    def __len__(self):
        return self.count

    @property
    def is_empty(self) -> bool:
        return self.count == 0

    def valid_entries(self) -> torch.Tensor:
        return self.memory_queue[: self.count]

    @torch.no_grad()
    def enqueue(self, embeddings: torch.Tensor):
        embeddings = F.normalize(embeddings.detach().to(self.device), p=2, dim=-1)
        if embeddings.size(-1) != self.embedding_dim:
            raise ValueError(f"Expected embedding dim {self.embedding_dim}, got {embeddings.size(-1)}")

        if embeddings.size(0) >= self.memory_size:
            self.memory_queue.copy_(embeddings[-self.memory_size :])
            self.ptr = 0
            self.count = self.memory_size
            return

        batch_size = embeddings.size(0)
        end = self.ptr + batch_size
        if end <= self.memory_size:
            self.memory_queue[self.ptr:end] = embeddings
        else:
            first = self.memory_size - self.ptr
            self.memory_queue[self.ptr:] = embeddings[:first]
            self.memory_queue[: end - self.memory_size] = embeddings[first:]

        self.ptr = end % self.memory_size
        self.count = min(self.memory_size, self.count + batch_size)

    def retrieve_neighbors(self, query: torch.Tensor, k: int = 10) -> torch.Tensor:
        if self.is_empty:
            return query.new_empty(query.size(0), 0, self.embedding_dim)

        query = F.normalize(query, p=2, dim=-1)
        entries = self.valid_entries()
        k = min(k, entries.size(0))
        scores = query @ entries.T
        indices = torch.topk(scores, k=k, dim=1).indices
        return entries[indices]
