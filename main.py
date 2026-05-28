import argparse
import logging
import os

from src.data.loader import get_dataloader
from src.evaluate import discover_checkpoints, load_smec_checkpoint, run_evaluation
from src.models.smec_wrapper import SMECModel
from src.trainer import SMECTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_dimensions(value: str | None, full_dim: int):
    if not value:
        return [full_dim, full_dim // 2, full_dim // 4]
    dims = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not dims:
        raise ValueError("--dimensions must contain at least one integer")
    if dims[0] != full_dim:
        dims = [full_dim] + dims
    return dims


def main():
    parser = argparse.ArgumentParser(description="SMEC: Sequential Matryoshka Embedding Compression")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train", help="Mode: train or eval")

    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Backbone model name")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint to evaluate")

    parser.add_argument("--dataset_name", type=str, default="quora", help="Dataset name/path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_length", type=int, default=128, help="Training tokenizer max sequence length")
    parser.add_argument("--eval_max_length", type=int, default=512, help="Evaluation tokenizer max sequence length")

    parser.add_argument("--epochs", type=int, default=3, help="Epochs per compressed dimension")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for ADS stages")
    parser.add_argument("--dimensions", type=str, default=None, help="Comma-separated dimensions, e.g. 768,384,192")
    parser.add_argument("--alpha", type=float, default=0.2, help="Weight for S-XBM similarity preservation loss")
    parser.add_argument("--memory_size", type=int, default=4096, help="S-XBM memory queue size")
    parser.add_argument("--top_k", type=int, default=10, help="Number of S-XBM neighbors per query")
    parser.add_argument("--ads_temperature", type=float, default=1.0, help="ADS Gumbel-Softmax temperature")
    parser.add_argument("--embedding_cache_dir", type=str, default=None, help="Reserved for future precomputed embedding cache")
    args = parser.parse_args()

    logger.info("Initializing model: %s", args.model_name)
    model = SMECModel(args.model_name, ads_temperature=args.ads_temperature)

    if args.mode == "train":
        train_loader = get_dataloader(
            args.dataset_name,
            batch_size=args.batch_size,
            split="train",
            max_length=args.max_length,
        )
        trainer = SMECTrainer(
            model=model,
            train_loader=train_loader,
            output_dir=args.output_dir,
            learning_rate=args.lr,
            max_length=args.max_length,
            alpha=args.alpha,
            memory_size=args.memory_size,
            top_k=args.top_k,
        )
        dimensions = parse_dimensions(args.dimensions, model.embedding_dim)
        logger.info("Training SMEC dimensions: %s", dimensions)
        trainer.train_sequential(dimensions, epochs_per_dim=args.epochs)
        return

    checkpoints_to_run = [args.checkpoint] if args.checkpoint else discover_checkpoints(args.output_dir)
    if not checkpoints_to_run:
        logger.warning("No checkpoints found. Evaluating frozen backbone only.")
        run_evaluation(
            model,
            tasks=["STSBenchmark"],
            output_folder=os.path.join("results", "results_base"),
            batch_size=args.batch_size,
            max_length=args.eval_max_length,
        )
        return

    for checkpoint_path in checkpoints_to_run:
        logger.info("Evaluating checkpoint: %s", checkpoint_path)
        eval_model = SMECModel(args.model_name, ads_temperature=args.ads_temperature)
        try:
            load_smec_checkpoint(eval_model, checkpoint_path)
            eval_subfolder = f"results_{os.path.basename(checkpoint_path)}"
            run_evaluation(
                eval_model,
                tasks=["STSBenchmark"],
                output_folder=os.path.join("results", eval_subfolder),
                batch_size=args.batch_size,
                max_length=args.eval_max_length,
            )
        except Exception as exc:
            logger.error("Failed to evaluate %s: %s", checkpoint_path, exc)


if __name__ == "__main__":
    main()
