import logging
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .loss import SMECLoss
from .models.memory import SelectiveCrossBatchMemory
from .models.smec_wrapper import SMECModel

logger = logging.getLogger(__name__)


class SMECTrainer:
    """Trainer for Sequential Matryoshka Embedding Compression."""

    def __init__(
        self,
        model: SMECModel,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        learning_rate: float = 2e-5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "./checkpoints",
        max_length: int = 128,
        alpha: float = 0.2,
        memory_size: int = 4096,
        top_k: int = 10,
    ):
        self.device = device
        self.model = model.to(device)
        self.model.freeze_backbone()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = output_dir
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.top_k = top_k
        self.criterion = SMECLoss(alpha=alpha, top_k=top_k)
        self.optimizer = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
        self.memory = self._new_memory()

        os.makedirs(output_dir, exist_ok=True)

    def _new_memory(self):
        return SelectiveCrossBatchMemory(
            memory_size=self.memory_size,
            embedding_dim=self.model.embedding_dim,
            device=self.device,
        )

    def train_one_epoch(self, epoch_idx: int, target_dim: int):
        self.model.freeze_backbone()
        self.model.freeze_completed_stages()
        total_loss = 0.0
        total_rank = 0.0
        total_preserve = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx} [Dim {target_dim}]")
        for batch in pbar:
            queries = self._tokenize(batch["queries"])
            positives = self._tokenize(batch["positives"])

            query_input_ids = queries["input_ids"].to(self.device)
            query_attention = queries["attention_mask"].to(self.device)
            positive_input_ids = positives["input_ids"].to(self.device)
            positive_attention = positives["attention_mask"].to(self.device)

            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                query_original = self.model.encode_original(query_input_ids, query_attention)
                positive_original = self.model.encode_original(positive_input_ids, positive_attention)
                query_compressed = self.model.compress_embeddings(query_original, target_dim=target_dim)
                positive_compressed = self.model.compress_embeddings(positive_original, target_dim=target_dim)

                loss, parts = self.criterion(
                    query_original=query_original,
                    positive_original=positive_original,
                    query_compressed=query_compressed,
                    positive_compressed=positive_compressed,
                    memory_queue=self.memory,
                    compress_fn=lambda x: self.model.compress_embeddings(x, target_dim=target_dim),
                )

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.memory.enqueue(positive_original)

            loss_value = float(loss.detach().cpu())
            rank_value = float(parts["rank_loss"].cpu())
            preserve_value = float(parts["preserve_loss"].cpu())
            total_loss += loss_value
            total_rank += rank_value
            total_preserve += preserve_value
            pbar.set_postfix({"loss": loss_value, "rank": rank_value, "preserve": preserve_value})

        steps = max(len(self.train_loader), 1)
        logger.info(
            "End Epoch %s [Dim %s] - loss=%.4f rank=%.4f preserve=%.4f",
            epoch_idx,
            target_dim,
            total_loss / steps,
            total_rank / steps,
            total_preserve / steps,
        )
        return total_loss / steps

    def train_sequential(self, dimensions: list[int], epochs_per_dim: int = 3):
        if not dimensions:
            raise ValueError("dimensions must not be empty")

        full_dim = self.model.embedding_dim
        if dimensions[0] != full_dim:
            dimensions = [full_dim] + dimensions

        self.model.reset_compression()
        for dim in dimensions:
            if dim == full_dim:
                logger.info("Saving frozen backbone baseline for dimension %s", dim)
                self.save_checkpoint(f"checkpoint_dim_{dim}")
                continue

            logger.info("=== Starting SMRL transition to dimension: %s ===", dim)
            self.model.ensure_stage(dim)
            self.model.to(self.device)
            self.model.freeze_completed_stages()
            self.memory = self._new_memory()

            trainable_params = self.model.trainable_compression_parameters()
            if not trainable_params:
                raise RuntimeError(f"No trainable ADS parameters for dimension {dim}")
            self.optimizer = optim.AdamW(trainable_params, lr=self.learning_rate)

            for epoch in range(epochs_per_dim):
                self.train_one_epoch(epoch, dim)

            self.save_checkpoint(f"checkpoint_dim_{dim}")

    def save_checkpoint(self, name: str):
        path = os.path.join(self.output_dir, name)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "metadata": self.model.checkpoint_metadata(),
        }
        torch.save(payload, path)
        logger.info("Saved model checkpoint to %s", path)

    def _tokenize(self, texts):
        from transformers import AutoTokenizer

        if not hasattr(self, "tokenizer"):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model.model_name)

        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
