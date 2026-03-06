#!/usr/bin/env python3
"""
Data preparation pipeline for F5-TTS training on Open Bible datasets.

Automates all steps: metadata creation, preprocessing, vocab verification,
training parameter estimation, and config generation.

Usage:
    # Single language
    python prepare_data.py --dataset-base /path/to/bible-tts-resources --languages Yoruba

    # Multiple languages
    python prepare_data.py --dataset-base /path/to/bible-tts-resources --languages Yoruba Ewe Hausa

    # Custom training settings
    python prepare_data.py \
        --dataset-base /path/to/bible-tts-resources \
        --languages Yoruba \
        --target-updates 500000 \
        --num-gpus 1 \
        --workers 4
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


SAMPLING_RATE = 24000
HOP_LENGTH = 256
FRAMES_PER_SECOND = SAMPLING_RATE / HOP_LENGTH  # 93.75

DEFAULT_TARGET_UPDATES = 500_000
DEFAULT_BATCH_SIZE_PER_GPU = 28_000
DEFAULT_MAX_SAMPLES = 32
DEFAULT_NUM_WORKERS = 4
DEFAULT_NUM_GPUS = 1
DEFAULT_GRAD_ACCUMULATION = 1
DEFAULT_WORKERS = 4

BASE_CONFIG = "F5TTS_v1_Base.yaml"
CONFIGS_DIR = Path("src/f5_tts/configs")


def slugify(language: str) -> str:
    """Convert language name to a dataset slug, e.g. 'Yoruba' -> 'open-bible-yoruba'."""
    return f"open-bible-{language.lower()}"


def create_metadata(language: str, dataset_base: str, data_dir: Path) -> pd.DataFrame:
    """Read train.tsv and write metadata.csv in the format expected by prepare_csv_wavs.py."""
    src = Path(dataset_base) / language
    tsv_path = src / "train.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"Training TSV not found: {tsv_path}")

    wav_dir = (src / "wav").resolve()
    if not wav_dir.exists():
        raise FileNotFoundError(f"Wav directory not found: {wav_dir}")

    train = pd.read_csv(tsv_path, sep="\t")
    train = train[["filename", "text"]]
    train.columns = ["audio_file", "text"]
    train["audio_file"] = train["audio_file"].apply(lambda x: str(wav_dir / x))

    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "metadata.csv"
    train.to_csv(csv_path, index=False, sep="|")
    print(f"  Metadata: {csv_path} ({len(train)} samples)")
    print(f"  Wav dir:  {wav_dir}")
    return train


def run_preprocessing(data_dir: Path, out_dir: Path, workers: int):
    """Run prepare_csv_wavs.py to produce raw.arrow, duration.json, vocab.txt."""
    csv_path = data_dir / "metadata.csv"
    cmd = [
        sys.executable,
        "src/f5_tts/train/datasets/prepare_csv_wavs.py",
        str(csv_path),
        str(out_dir),
        "--pretrain",
        "--workers",
        str(workers),
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"  Preprocessed data saved to: {out_dir}")


def verify_vocab(out_dir: Path) -> list[str]:
    """Load and display the generated vocabulary."""
    vocab_path = out_dir / "vocab.txt"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    with open(vocab_path) as f:
        vocab = [line.rstrip("\n") for line in f]

    print(f"  Vocab size: {len(vocab)}")
    if vocab:
        print(f"  First entry is space: {repr(vocab[0]) == repr(' ')}")
    return vocab


def load_durations(out_dir: Path) -> list[float]:
    """Load duration.json and print summary stats."""
    dur_path = out_dir / "duration.json"
    if not dur_path.exists():
        raise FileNotFoundError(f"Duration file not found: {dur_path}")

    with open(dur_path) as f:
        durations = json.load(f)["duration"]

    total_h = sum(durations) / 3600
    avg_s = sum(durations) / len(durations)
    print(f"  Samples: {len(durations)}")
    print(f"  Total: {total_h:.2f} hours")
    print(f"  Duration: avg={avg_s:.2f}s, min={min(durations):.2f}s, max={max(durations):.2f}s")
    return durations


def estimate_training_params(
    durations: list[float],
    batch_size_per_gpu: int,
    max_samples: int,
    num_gpus: int,
    grad_accumulation_steps: int,
    target_updates: int,
) -> dict:
    """Simulate DynamicBatchSampler to estimate updates/epoch and required epochs."""
    frame_lens = sorted([d * FRAMES_PER_SECOND for d in durations])

    num_batches = 0
    batch_frames = 0
    batch_count = 0
    for fl in frame_lens:
        if batch_frames + fl <= batch_size_per_gpu and (max_samples == 0 or batch_count < max_samples):
            batch_frames += fl
            batch_count += 1
        else:
            if batch_count > 0:
                num_batches += 1
            if fl <= batch_size_per_gpu:
                batch_frames = fl
                batch_count = 1
            else:
                batch_frames = 0
                batch_count = 0
    if batch_count > 0:
        num_batches += 1

    total_frames = sum(durations) * FRAMES_PER_SECOND
    updates_per_epoch = math.ceil(num_batches / (num_gpus * grad_accumulation_steps))
    epochs = math.ceil(target_updates / updates_per_epoch)
    utilization = total_frames / num_batches / batch_size_per_gpu * 100

    result = {
        "num_batches": num_batches,
        "updates_per_epoch": updates_per_epoch,
        "epochs": epochs,
        "avg_batch_utilization": utilization,
    }

    print(f"  Batches per epoch: {num_batches}")
    print(f"  Avg batch utilization: {utilization:.1f}%")
    print(f"  Updates per epoch: {updates_per_epoch}")
    print(f"  Target updates: {target_updates} -> epochs: {epochs}")
    return result


def generate_config(
    slug: str,
    epochs: int,
    batch_size_per_gpu: int,
    max_samples: int,
    num_workers: int,
) -> Path:
    """Generate a training YAML config from the base config."""
    base_path = CONFIGS_DIR / BASE_CONFIG
    with open(base_path) as f:
        config = yaml.safe_load(f)

    config["datasets"]["name"] = slug
    config["datasets"]["batch_size_per_gpu"] = batch_size_per_gpu
    config["datasets"]["max_samples"] = max_samples
    config["datasets"]["num_workers"] = num_workers

    config["optim"]["epochs"] = epochs

    config["ckpts"]["logger"] = "tensorboard"
    config["ckpts"]["save_per_updates"] = 10_000
    config["ckpts"]["keep_last_n_checkpoints"] = 5
    # Remove wandb keys if present
    config["ckpts"].pop("wandb_project", None)
    config["ckpts"].pop("wandb_run_name", None)
    config["ckpts"].pop("wandb_resume_id", None)

    config_name = f"F5TTS_v1_Base_{slug.replace('-', '_').title().replace('_', '_')}.yaml"
    # Produce a cleaner name: F5TTS_v1_Base_Open_Bible_{Language}.yaml
    lang = slug.replace("open-bible-", "").title()
    config_name = f"F5TTS_v1_Base_Open_Bible_{lang}.yaml"
    config_path = CONFIGS_DIR / config_name

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # Re-read and add the hydra header (yaml.dump doesn't preserve it well)
    rewrite_config_with_hydra_header(config_path, slug, epochs, batch_size_per_gpu, max_samples, num_workers)

    print(f"  Config: {config_path}")
    return config_path


def rewrite_config_with_hydra_header(
    config_path: Path,
    slug: str,
    epochs: int,
    batch_size_per_gpu: int,
    max_samples: int,
    num_workers: int,
):
    """Write a clean config with proper hydra interpolation syntax."""
    base_path = CONFIGS_DIR / BASE_CONFIG
    with open(base_path) as f:
        lines = f.readlines()

    replacements = {
        "  name: Emilia_ZH_EN": f"  name: {slug}",
        "  batch_size_per_gpu: 38400": f"  batch_size_per_gpu: {batch_size_per_gpu}",
        "  max_samples: 64": f"  max_samples: {max_samples}",
        "  num_workers: 16": f"  num_workers: {num_workers}",
        "  epochs: 11": f"  epochs: {epochs}",
        "  logger: wandb": "  logger: tensorboard",
        "  save_per_updates: 50000": "  save_per_updates: 10000",
        "  keep_last_n_checkpoints: -1": "  keep_last_n_checkpoints: 5",
    }
    skip_keys = ["wandb_project:", "wandb_run_name:", "wandb_resume_id:"]

    new_lines = []
    for line in lines:
        stripped = line.strip()
        if any(k in stripped for k in skip_keys):
            continue
        replaced = False
        for old, new in replacements.items():
            if old in line:
                new_lines.append(line.replace(old, new))
                replaced = True
                break
        if not replaced:
            new_lines.append(line)

    with open(config_path, "w") as f:
        f.writelines(new_lines)


def prepare_language(
    language: str,
    dataset_base: str,
    target_updates: int,
    batch_size_per_gpu: int,
    max_samples: int,
    num_gpus: int,
    grad_accumulation_steps: int,
    num_workers: int,
    workers: int,
    skip_preprocess: bool,
):
    """Full preparation pipeline for one language."""
    slug = slugify(language)
    data_dir = Path("data") / slug
    out_dir = Path("data") / f"{slug}_pinyin"

    print(f"\n{'='*60}")
    print(f"Preparing: {language} ({slug})")
    print(f"{'='*60}")

    print("\n[1/5] Creating metadata CSV...")
    create_metadata(language, dataset_base, data_dir)

    if not skip_preprocess:
        print("\n[2/5] Running preprocessing (prepare_csv_wavs.py)...")
        run_preprocessing(data_dir, out_dir, workers)
    else:
        print("\n[2/5] Skipping preprocessing (--skip-preprocess)")
        if not out_dir.exists():
            print(f"  WARNING: {out_dir} does not exist. Run without --skip-preprocess first.")
            return

    print("\n[3/5] Verifying vocabulary...")
    verify_vocab(out_dir)

    print("\n[4/5] Computing training parameters...")
    durations = load_durations(out_dir)
    params = estimate_training_params(
        durations, batch_size_per_gpu, max_samples, num_gpus, grad_accumulation_steps, target_updates
    )

    print("\n[5/5] Generating training config...")
    config_path = generate_config(slug, params["epochs"], batch_size_per_gpu, max_samples, num_workers)

    print(f"\n{'─'*60}")
    print(f"Done: {language}")
    print(f"  Data dir:    {data_dir}")
    print(f"  Prepared:    {out_dir}")
    print(f"  Config:      {config_path}")
    print(f"  Train cmd:   accelerate launch --mixed_precision bf16 src/f5_tts/train/train.py --config-name {config_path.name}")
    print(f"{'─'*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Open Bible TTS data for F5-TTS training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        required=True,
        help="Language names matching directories in the dataset base path (e.g. Yoruba Ewe Hausa)",
    )
    parser.add_argument(
        "--dataset-base",
        required=True,
        help="Base path containing language directories (e.g. /path/to/bible-tts-resources)",
    )
    parser.add_argument(
        "--target-updates",
        type=int,
        default=DEFAULT_TARGET_UPDATES,
        help=f"Target number of training updates (default: {DEFAULT_TARGET_UPDATES})",
    )
    parser.add_argument(
        "--batch-size-per-gpu",
        type=int,
        default=DEFAULT_BATCH_SIZE_PER_GPU,
        help=f"Frame budget per GPU batch (default: {DEFAULT_BATCH_SIZE_PER_GPU})",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Max sequences per batch (default: {DEFAULT_MAX_SAMPLES})",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=DEFAULT_NUM_GPUS,
        help=f"Number of GPUs for training (default: {DEFAULT_NUM_GPUS})",
    )
    parser.add_argument(
        "--grad-accumulation-steps",
        type=int,
        default=DEFAULT_GRAD_ACCUMULATION,
        help=f"Gradient accumulation steps (default: {DEFAULT_GRAD_ACCUMULATION})",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"Dataloader workers for training config (default: {DEFAULT_NUM_WORKERS})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Threads for audio preprocessing (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip the prepare_csv_wavs.py step (use if already preprocessed)",
    )
    args = parser.parse_args()

    available = [d.name for d in Path(args.dataset_base).iterdir() if d.is_dir()]
    for lang in args.languages:
        if lang not in available:
            print(f"ERROR: Language '{lang}' not found in {args.dataset_base}")
            print(f"  Available: {', '.join(sorted(available))}")
            sys.exit(1)

    for lang in args.languages:
        prepare_language(
            language=lang,
            dataset_base=args.dataset_base,
            target_updates=args.target_updates,
            batch_size_per_gpu=args.batch_size_per_gpu,
            max_samples=args.max_samples,
            num_gpus=args.num_gpus,
            grad_accumulation_steps=args.grad_accumulation_steps,
            num_workers=args.num_workers,
            workers=args.workers,
            skip_preprocess=args.skip_preprocess,
        )

    if len(args.languages) > 1:
        print(f"\n{'='*60}")
        print(f"All {len(args.languages)} languages prepared successfully.")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
