#!/usr/bin/env python3
"""
Data preparation pipeline for F5-TTS training on Open Bible datasets.

Automates all steps: metadata creation, preprocessing, vocab verification,
training parameter estimation, and config generation.

Usage:
    # Single language (monolingual, custom tokenizer)
    python prepare_data.py --languages Yoruba

    # Single language with pinyin tokenizer
    python prepare_data.py --languages Yoruba --tokenizer pinyin

    # Multiple languages (each trained separately)
    python prepare_data.py --languages Yoruba Ewe Hausa

    # Multilingual (combine languages into one dataset)
    python prepare_data.py --languages Swahili Chichewa Matengo --multilingual

    # Use a local dataset directory instead
    python prepare_data.py --dataset-base /path/to/bible-tts-resources --languages Yoruba

    # Custom training settings
    python prepare_data.py \
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
from io import BytesIO
from pathlib import Path

import pandas as pd
import soundfile as sf
import yaml
from tqdm import tqdm


SAMPLING_RATE = 24000
HOP_LENGTH = 256
FRAMES_PER_SECOND = SAMPLING_RATE / HOP_LENGTH  # 93.75

DEFAULT_HF_REPO = "davidguzmanr/open-bible-resources"
DEFAULT_TARGET_UPDATES = 500_000
DEFAULT_BATCH_SIZE_PER_GPU = 28_000
DEFAULT_MAX_SAMPLES = 32
DEFAULT_NUM_WORKERS = 4
DEFAULT_NUM_GPUS = 1
DEFAULT_GRAD_ACCUMULATION = 1
DEFAULT_WORKERS = 4
DEFAULT_MAX_DURATION = 15.0

BASE_CONFIG = "F5TTS_v1_Base.yaml"
CONFIGS_DIR = Path("src/f5_tts/configs")


TESTAMENT_SUFFIX = {
    "New Testament": "nt",
    "Old Testament": "ot",
}


def slugify(language: str, testament: str | None = None) -> str:
    """Convert language name to a dataset slug, e.g. 'Yoruba' -> 'open-bible-yoruba'.
    If a testament filter is provided, appends a suffix, e.g. 'open-bible-yoruba-nt'."""
    slug = f"open-bible-{language.lower()}"
    if testament:
        suffix = TESTAMENT_SUFFIX.get(testament, testament.lower().replace(" ", "-"))
        slug = f"{slug}-{suffix}"
    return slug


def multilingual_slug(languages: list[str], testament_filters: dict[str, str] | None = None) -> str:
    """Create a slug for multilingual dataset, reflecting any testament filters."""
    parts = []
    for lang in sorted(languages):
        testament = (testament_filters or {}).get(lang)
        if testament:
            suffix = TESTAMENT_SUFFIX.get(testament, testament.lower().replace(" ", "-"))
            parts.append(f"{lang.lower()}-{suffix}")
        else:
            parts.append(lang.lower())
    return "open-bible-" + "-".join(parts)


def create_metadata(language: str, dataset_base: str, data_dir: Path, max_duration: float, testament_filter: str | None = None) -> pd.DataFrame:
    """Read train.tsv and write metadata.csv in the format expected by prepare_csv_wavs.py."""
    src = Path(dataset_base) / language
    tsv_path = src / "train.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"Training TSV not found: {tsv_path}")

    wav_dir = (src / "wav").resolve()
    if not wav_dir.exists():
        raise FileNotFoundError(f"Wav directory not found: {wav_dir}")

    train = pd.read_csv(tsv_path, sep="\t")

    if testament_filter and "testament" in train.columns:
        total_before = len(train)
        train = train[train["testament"] == testament_filter].reset_index(drop=True)
        print(f"  Testament filter '{testament_filter}': {len(train)}/{total_before} samples kept")

    train = train[["filename", "text"]]
    train.columns = ["audio_file", "text"]
    train["audio_file"] = train["audio_file"].apply(lambda x: str(wav_dir / x))

    total_before = len(train)
    if max_duration > 0:
        durations = train["audio_file"].apply(lambda p: sf.info(p).duration)
        train = train[durations <= max_duration].reset_index(drop=True)
        n_filtered = total_before - len(train)
        if n_filtered > 0:
            print(f"  Filtered {n_filtered}/{total_before} samples exceeding {max_duration}s")

    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "metadata.csv"
    train.to_csv(csv_path, index=False, sep="|")
    print(f"  Metadata: {csv_path} ({len(train)} samples)")
    print(f"  Wav dir:  {wav_dir}")
    return train


def download_from_huggingface(language: str, hf_repo: str, data_dir: Path, max_duration: float, testament_filter: str | None = None) -> pd.DataFrame:
    """Download a language dataset from Hugging Face and write wav files + metadata.csv."""
    from datasets import Audio, load_dataset

    print(f"  Loading {hf_repo}/{language} from Hugging Face...")
    ds = load_dataset(hf_repo, language)

    wavs_dir = data_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    subset = ds["train"].cast_column("audio", Audio(decode=False))

    rows = []
    n_filtered = 0
    n_testament_filtered = 0
    for i in tqdm(range(len(subset)), desc=f"Downloading {language} wav files"):
        sample = subset[i]

        if testament_filter and sample["testament"] != testament_filter:
            n_testament_filtered += 1
            continue

        testament = sample["testament"].replace(" ", "-")
        book = sample["book"].replace(" ", "-")
        chapter = sample["chapter"]
        verse = sample["verse"]
        filename = f"{testament}-{book}-{chapter}-{verse}.wav"

        wav_path = wavs_dir / filename
        audio_data, sr = sf.read(BytesIO(sample["audio"]["bytes"]))

        if max_duration > 0 and len(audio_data) / sr > max_duration:
            n_filtered += 1
            continue

        if not wav_path.exists():
            sf.write(str(wav_path), audio_data, sr)

        rows.append({"audio_file": str(wav_path.resolve()), "text": sample["text"]})

    if n_testament_filtered > 0:
        print(f"  Testament filter '{testament_filter}': {len(subset) - n_testament_filtered}/{len(subset)} samples kept")
    if n_filtered > 0:
        print(f"  Filtered {n_filtered}/{len(subset)} samples exceeding {max_duration}s")

    metadata = pd.DataFrame(rows)
    csv_path = data_dir / "metadata.csv"
    metadata.to_csv(csv_path, index=False, sep="|")
    print(f"  Downloaded {len(rows)} samples to {wavs_dir}")
    print(f"  Metadata: {csv_path}")
    return metadata


def run_preprocessing(data_dir: Path, out_dir: Path, workers: int, tokenizer: str):
    """Run prepare_csv_wavs.py to produce raw.arrow, duration.json, vocab.txt."""
    csv_path = data_dir / "metadata.csv"
    cmd = [
        sys.executable,
        "src/f5_tts/train/datasets/prepare_csv_wavs.py",
        str(csv_path),
        str(out_dir),
        "--workers",
        str(workers),
    ]
    if tokenizer == "custom":
        cmd.append("--pretrain")
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
    single_gpu_updates_per_epoch = math.ceil(num_batches / grad_accumulation_steps)
    multi_gpu_updates_per_epoch = math.ceil(num_batches / (num_gpus * grad_accumulation_steps))
    epochs = math.ceil(target_updates / single_gpu_updates_per_epoch)
    utilization = total_frames / num_batches / batch_size_per_gpu * 100

    result = {
        "num_batches": num_batches,
        "updates_per_epoch": multi_gpu_updates_per_epoch,
        "epochs": epochs,
        "avg_batch_utilization": utilization,
    }

    print(f"  Batches per epoch: {num_batches}")
    print(f"  Avg batch utilization: {utilization:.1f}%")
    print(f"  Updates/epoch (1 GPU): {single_gpu_updates_per_epoch}")
    print(f"  Updates/epoch ({num_gpus} GPUs): {multi_gpu_updates_per_epoch}")
    print(f"  Target updates (1-GPU equivalent): {target_updates} -> epochs: {epochs}")
    if num_gpus > 1:
        actual_updates = multi_gpu_updates_per_epoch * epochs
        print(f"  Actual optimizer updates with {num_gpus} GPUs: ~{actual_updates} (effective batch size {num_gpus}x)")
    return result


def generate_config(
    slug: str,
    epochs: int,
    batch_size_per_gpu: int,
    max_samples: int,
    num_workers: int,
    tokenizer: str,
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
    config["ckpts"].pop("wandb_project", None)
    config["ckpts"].pop("wandb_run_name", None)
    config["ckpts"].pop("wandb_resume_id", None)

    lang = slug.replace("open-bible-", "").title()
    config_name = f"F5TTS_v1_Base_Open_Bible_{lang}.yaml"
    config_path = CONFIGS_DIR / config_name

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    rewrite_config_with_hydra_header(config_path, slug, epochs, batch_size_per_gpu, max_samples, num_workers, tokenizer)

    print(f"  Config: {config_path}")
    return config_path


def rewrite_config_with_hydra_header(
    config_path: Path,
    slug: str,
    epochs: int,
    batch_size_per_gpu: int,
    max_samples: int,
    num_workers: int,
    tokenizer: str,
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

    if tokenizer == "custom":
        vocab_path = f"data/{slug}_custom/vocab.txt"
        replacements["  tokenizer: pinyin"] = "  tokenizer: custom"
        replacements["  tokenizer_path: null"] = f"  tokenizer_path: {vocab_path}"

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
    dataset_base: str | None,
    hf_repo: str,
    max_duration: float,
    target_updates: int,
    batch_size_per_gpu: int,
    max_samples: int,
    num_gpus: int,
    grad_accumulation_steps: int,
    num_workers: int,
    workers: int,
    skip_preprocess: bool,
    tokenizer: str,
    testament_filter: str | None = None,
):
    """Full preparation pipeline for one language."""
    slug = slugify(language, testament_filter)
    data_dir = Path("data") / slug
    out_dir = Path("data") / f"{slug}_{tokenizer}"

    print(f"\n{'='*60}")
    print(f"Preparing: {language} ({slug})")
    print(f"Tokenizer: {tokenizer}")
    if testament_filter:
        print(f"Testament filter: {testament_filter}")
    print(f"{'='*60}")

    if dataset_base:
        print("\n[1/5] Creating metadata CSV from local data...")
        create_metadata(language, dataset_base, data_dir, max_duration, testament_filter)
    else:
        print(f"\n[1/5] Downloading from Hugging Face ({hf_repo})...")
        download_from_huggingface(language, hf_repo, data_dir, max_duration, testament_filter)

    if not skip_preprocess:
        print("\n[2/5] Running preprocessing (prepare_csv_wavs.py)...")
        run_preprocessing(data_dir, out_dir, workers, tokenizer)
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
    config_path = generate_config(slug, params["epochs"], batch_size_per_gpu, max_samples, num_workers, tokenizer)

    print(f"\n{'─'*60}")
    print(f"Done: {language}")
    print(f"  Data dir:    {data_dir}")
    print(f"  Prepared:    {out_dir}")
    print(f"  Config:      {config_path}")
    print(f"  Train cmd:   accelerate launch --mixed_precision bf16 src/f5_tts/train/train.py --config-name {config_path.name}")
    print(f"{'─'*60}")


def prepare_multilingual(
    languages: list[str],
    dataset_base: str | None,
    hf_repo: str,
    max_duration: float,
    target_updates: int,
    batch_size_per_gpu: int,
    max_samples: int,
    num_gpus: int,
    grad_accumulation_steps: int,
    num_workers: int,
    workers: int,
    skip_preprocess: bool,
    tokenizer: str,
    testament_filters: dict[str, str] | None = None,
):
    """Full preparation pipeline for multilingual training."""
    slug = multilingual_slug(languages, testament_filters)
    combined_data_dir = Path("data") / slug
    out_dir = Path("data") / f"{slug}_{tokenizer}"

    print(f"\n{'='*60}")
    print(f"Preparing multilingual: {', '.join(languages)}")
    print(f"Slug: {slug}")
    print(f"Tokenizer: {tokenizer}")
    if testament_filters:
        print(f"Testament filters: {testament_filters}")
    print(f"{'='*60}")

    # Step 1: Download/prepare each language
    print(f"\n[1/6] Downloading {len(languages)} languages...")
    all_metadata = []
    for language in languages:
        testament_filter = (testament_filters or {}).get(language)
        lang_data_dir = Path("data") / slugify(language, testament_filter)
        print(f"\n  --- {language} ---")
        if dataset_base:
            metadata = create_metadata(language, dataset_base, lang_data_dir, max_duration, testament_filter)
        else:
            metadata = download_from_huggingface(language, hf_repo, lang_data_dir, max_duration, testament_filter)
        all_metadata.append(metadata)

    # Step 2: Combine metadata CSVs
    print(f"\n[2/6] Combining metadata from {len(languages)} languages...")
    combined = pd.concat(all_metadata, ignore_index=True)
    combined_data_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = combined_data_dir / "metadata.csv"
    combined.to_csv(combined_csv, index=False, sep="|")
    print(f"  Combined metadata: {combined_csv} ({len(combined)} samples)")
    for lang, meta in zip(languages, all_metadata):
        print(f"    {lang}: {len(meta)} samples")

    # Step 3: Preprocess combined dataset
    if not skip_preprocess:
        print("\n[3/6] Running preprocessing on combined dataset...")
        run_preprocessing(combined_data_dir, out_dir, workers, tokenizer)
    else:
        print("\n[3/6] Skipping preprocessing (--skip-preprocess)")
        if not out_dir.exists():
            print(f"  WARNING: {out_dir} does not exist. Run without --skip-preprocess first.")
            return

    # Step 4: Verify vocab
    print("\n[4/6] Verifying vocabulary...")
    verify_vocab(out_dir)

    # Step 5: Compute training parameters
    print("\n[5/6] Computing training parameters...")
    durations = load_durations(out_dir)
    params = estimate_training_params(
        durations, batch_size_per_gpu, max_samples, num_gpus, grad_accumulation_steps, target_updates
    )

    # Step 6: Generate config
    print("\n[6/6] Generating training config...")
    config_path = generate_config(slug, params["epochs"], batch_size_per_gpu, max_samples, num_workers, tokenizer)

    print(f"\n{'─'*60}")
    print(f"Done: Multilingual ({', '.join(languages)})")
    print(f"  Languages:   {', '.join(languages)}")
    print(f"  Data dir:    {combined_data_dir}")
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
        "--multilingual",
        action="store_true",
        help="Combine all languages into one multilingual dataset for training",
    )
    parser.add_argument(
        "--tokenizer",
        choices=["custom", "pinyin"],
        default="custom",
        help="Tokenizer type: 'custom' builds vocab from dataset, 'pinyin' uses pretrained Chinese/English vocab (default: custom)",
    )
    parser.add_argument(
        "--dataset-base",
        default=None,
        help="Base path containing language directories. If not provided, data is downloaded from Hugging Face.",
    )
    parser.add_argument(
        "--hf-repo",
        default=DEFAULT_HF_REPO,
        help=f"Hugging Face dataset repo (default: {DEFAULT_HF_REPO}). Used when --dataset-base is not set.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=DEFAULT_MAX_DURATION,
        help=f"Discard audio files longer than this many seconds; 0 to disable (default: {DEFAULT_MAX_DURATION})",
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
    parser.add_argument(
        "--filter-testament",
        type=str,
        default=None,
        help=(
            'JSON dict mapping language names to testament strings, e.g. \'{"Yoruba": "New Testament"}\'. '
            "Languages not listed are not filtered. Default: use all data."
        ),
    )
    args = parser.parse_args()

    testament_filters: dict[str, str] | None = None
    if args.filter_testament:
        try:
            testament_filters = json.loads(args.filter_testament)
        except json.JSONDecodeError as e:
            print(f"ERROR: --filter-testament is not valid JSON: {e}")
            sys.exit(1)

    if args.dataset_base:
        base = Path(args.dataset_base)
        if not base.is_dir():
            print(f"ERROR: --dataset-base directory not found: {base}")
            sys.exit(1)
        available = [d.name for d in base.iterdir() if d.is_dir()]
        for lang in args.languages:
            if lang not in available:
                print(f"ERROR: Language '{lang}' not found in {args.dataset_base}")
                print(f"  Available: {', '.join(sorted(available))}")
                sys.exit(1)

    if args.multilingual:
        if len(args.languages) < 2:
            print("ERROR: --multilingual requires at least 2 languages")
            sys.exit(1)
        prepare_multilingual(
            languages=args.languages,
            dataset_base=args.dataset_base,
            hf_repo=args.hf_repo,
            max_duration=args.max_duration,
            target_updates=args.target_updates,
            batch_size_per_gpu=args.batch_size_per_gpu,
            max_samples=args.max_samples,
            num_gpus=args.num_gpus,
            grad_accumulation_steps=args.grad_accumulation_steps,
            num_workers=args.num_workers,
            workers=args.workers,
            skip_preprocess=args.skip_preprocess,
            tokenizer=args.tokenizer,
            testament_filters=testament_filters,
        )
    else:
        for lang in args.languages:
            prepare_language(
                language=lang,
                dataset_base=args.dataset_base,
                hf_repo=args.hf_repo,
                max_duration=args.max_duration,
                target_updates=args.target_updates,
                batch_size_per_gpu=args.batch_size_per_gpu,
                max_samples=args.max_samples,
                num_gpus=args.num_gpus,
                grad_accumulation_steps=args.grad_accumulation_steps,
                num_workers=args.num_workers,
                workers=args.workers,
                skip_preprocess=args.skip_preprocess,
                tokenizer=args.tokenizer,
                testament_filter=(testament_filters or {}).get(lang),
            )

        if len(args.languages) > 1:
            print(f"\n{'='*60}")
            print(f"All {len(args.languages)} languages prepared successfully.")
            print(f"{'='*60}")


if __name__ == "__main__":
    main()
