import logging
import os
from typing import List

import mteb
import torch
from mteb import MTEB

logger = logging.getLogger(__name__)


class SMECModelWrapper(mteb.EncoderProtocol):
    """MTEB-compatible wrapper around a frozen-backbone SMEC model."""

    def __init__(self, smec_model, batch_size=32, device=None, max_length=512):
        self.model = smec_model
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.model.to(self.device)
        self.model.eval()

    @property
    def mteb_model_meta(self):
        from mteb.models.model_meta import ModelMeta

        return ModelMeta.model_construct(
            name="SMEC-Model",
            revision="1.0.0",
            release_date="2026-02-17",
            languages=["eng-Latn"],
            loader=None,
            max_tokens=self.max_length,
            embed_dim=self.model.current_dim,
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            framework=["PyTorch"],
            similarity_fn_name="cosine",
            use_instructions=False,
            training_datasets=set(),
            adapted_from=None,
            superseded_by=None,
            modalities=["text"],
            model_type=["dense"],
            citation=None,
            contacts=None,
            reference=None,
        )

    def encode(self, sentences: List[str], batch_size: int = 32, **kwargs):
        if not isinstance(sentences, list):
            sentences = list(sentences)

        if sentences and not isinstance(sentences[0], str):
            flattened = []
            for item in sentences:
                if isinstance(item, dict) and "text" in item:
                    text = item["text"]
                    flattened.extend(text if isinstance(text, list) else [text])
                elif isinstance(item, (list, tuple)):
                    flattened.extend(item)
                elif isinstance(item, str):
                    flattened.append(item)
            sentences = flattened

        from transformers import AutoTokenizer

        if not hasattr(self, "tokenizer"):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model.model_name)

        effective_batch_size = batch_size or self.batch_size
        all_embeddings = []
        for i in range(0, len(sentences), effective_batch_size):
            batch_texts = sentences[i : i + effective_batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            ).to(self.device)

            with torch.no_grad():
                embeddings = self.model(inputs["input_ids"], inputs["attention_mask"])
                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0).numpy()

    def similarity(self, embeddings1, embeddings2):
        import numpy as np

        return np.dot(embeddings1, embeddings2.T)

    def similarity_pairwise(self, embeddings1, embeddings2):
        import numpy as np

        return np.multiply(embeddings1, embeddings2).sum(axis=1)


def load_smec_checkpoint(model, checkpoint_path: str):
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        metadata = payload.get("metadata", {})
        model.load_checkpoint_metadata(metadata)
        model.load_state_dict(payload["model_state_dict"])
        return metadata

    # Compatibility with the original project checkpoints.
    if "checkpoint_dim_" in checkpoint_path:
        dim_str = checkpoint_path.split("checkpoint_dim_")[-1]
        if dim_str.isdigit():
            model.set_ads_target_dim(int(dim_str))

    if "ads.gate_logits" in payload:
        payload = dict(payload)
        payload["compression_stages.0.gate_logits"] = payload.pop("ads.gate_logits")

    model.load_state_dict(payload, strict=False)
    return model.checkpoint_metadata()


def run_evaluation(model, tasks=None, output_folder="results", batch_size=32, max_length=512):
    tasks = tasks or ["QuoraRetrieval"]
    wrapper = SMECModelWrapper(model, batch_size=batch_size, max_length=max_length)
    task_objs = mteb.get_tasks(tasks=tasks)
    evaluation = MTEB(tasks=task_objs)
    results = evaluation.run(wrapper, output_folder=output_folder)
    logger.info("Evaluation Results: %s", results)
    return results


def discover_checkpoints(output_dir: str):
    if not os.path.exists(output_dir):
        return []

    checkpoints = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if "checkpoint_dim_" in f]

    def dim_key(path):
        dim_str = path.split("checkpoint_dim_")[-1]
        return int(dim_str) if dim_str.isdigit() else 0

    return sorted(checkpoints, key=dim_key, reverse=True)

