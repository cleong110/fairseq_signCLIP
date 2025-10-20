import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
from tqdm import tqdm

from search_with_text_and_poses import embed_pose, load_pose, MAX_FRAMES_DEFAULT

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def infer_segments_json(pose_path: Path, autoseg_model: str) -> Path:
    """
    Infer autosegmented JSON path from pose path and autoseg model name.
    Example:
      video.pose-mediapipe.pose -> video.model_{autoseg_model}.autosegmented_segments.json
    """
    base = Path(".".join(pose_path.name.split(".")[:-2]))  # strip .pose-mediapipe.pose
    return pose_path.with_name(
        f"{base}.model_{autoseg_model}.autosegmented_segments.json"
    )


def load_segments(json_path: Path) -> list[dict]:
    """
    Load autosegmented segments JSON.
    Returns a list of dicts with keys:
        - type (SIGN or SENTENCE)
        - start_ms, end_ms, text
        - index (position within its type list)
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    segments = []
    for seg_type, entries in raw.items():
        logger.info(f"Reading segment type: {seg_type}")
        for idx, entry in enumerate(entries):
            segments.append(
                {
                    "type": seg_type,
                    "segment_index": idx,
                    "start_ms": entry["start_ms"],
                    "end_ms": entry["end_ms"],
                    "text": entry.get("text", ""),
                }
            )
    return segments


def get_existing_segments(
    dataset_dir: Path, model: str, video_stem: str, columns=None
) -> pd.DataFrame:
    """Return DataFrame of already-embedded segments for given model/video."""
    if not dataset_dir.exists() or not any(dataset_dir.rglob("*.parquet")):
        # ensure correct schema even if no data
        return pd.DataFrame(
            columns=[
                "video_stem",
                "type",
                "segment_index",
                "start_ms",
                "end_ms",
                "text",
                "embedding",
                "model",
            ]
        )

    dataset = ds.dataset(dataset_dir, format="parquet", partitioning="hive")
    filt = (ds.field("model") == model) & (ds.field("video_stem") == video_stem)

    table = dataset.to_table(filter=filt, columns=columns)
    return table.to_pandas()


def embed_autosegmented(
    pose_path: Path,
    autoseg_model: str,
    embedding_model: str,
    segments_json: Path | None,
    output_dir: Path,
) -> None:
    """Embed autosegmented segments and save parquet with hive partitioning."""

    video_stem = pose_path.stem

    langcode = pose_path.parent.parent.name
    assert len(langcode) == 3
    bible_name = pose_path.parent.name

    logger.info(f"Language code: {langcode}. Bible Name: {bible_name}")
    if segments_json is None:
        segments_json = infer_segments_json(pose_path, autoseg_model)

    output_dir = output_dir / langcode / bible_name
    assert output_dir.is_dir()
    logger.info(f"New Output dir: {output_dir}")

    if not segments_json.exists():
        raise FileNotFoundError(
            f"Segments JSON not found: {segments_json}, even though we have {pose_path}"
        )

    logger.info(f"Loading segments from {segments_json}")
    segments = load_segments(segments_json)

    existing = get_existing_segments(
        dataset_dir=output_dir,
        model=embedding_model,
        video_stem=video_stem,
        columns=["type", "segment_index", "start_ms", "end_ms"],
    )
    logger.info(f"Loaded existing: {existing}")

    existing_set = set(
        zip(
            existing["type"],
            existing["segment_index"],
            existing["start_ms"],
            existing["end_ms"],
        )
    )

    rows = []
    for seg in tqdm(segments, desc=f"Embedding {video_stem}"):
        key = (seg["type"], seg["segment_index"], seg["start_ms"], seg["end_ms"])
        if key in existing_set:
            continue

        pose = load_pose(
            pose_path=pose_path,
            start_time_ms=seg["start_ms"],
            end_time_ms=seg["end_ms"],
        )
        try:
            embedding = embed_pose(pose, model_name=embedding_model).squeeze()
        except AssertionError as e:
            if "Video too long" in str(e):
                logger.warning(
                    f"Skipping segment {key} for {video_stem}: "
                    f"frame count exceeds MAX_FRAMES ({MAX_FRAMES_DEFAULT})"
                )
                continue
            raise

        rows.append(
            {
                "video_stem": video_stem,
                "type": seg["type"],
                "segment_index": seg["segment_index"],
                "start_ms": seg["start_ms"],
                "end_ms": seg["end_ms"],
                "text": seg["text"],
                "embedding": embedding.tolist(),
                "model": embedding_model,  # partitioning key
            }
        )

    if not rows:
        logger.info(f"No new segments to embed for {video_stem}")
        return

    df = pd.DataFrame(rows)
    df.to_parquet(
        output_dir,
        index=False,
        engine="pyarrow",
        partition_cols=["model", "type", "video_stem"],
    )

    logger.info(f"Saved {len(df)} new embeddings to dataset at {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute pose embeddings for autosegmented segments."
    )
    parser.add_argument("pose_path", type=Path, help="Input .pose-mediapipe.pose file")
    parser.add_argument("--segments_json", type=Path, default=None)
    parser.add_argument(
        "--autoseg_model",
        type=str,
        default="e4s-1",
        help="Name of the model used to generate autosegmentation JSON",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="asl_finetune_checkpoint_best",
        help="Name of the model used to generate embeddings",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(
            "/data/petabyte/cleong/data/DBL_Deaf_Bibles/autosegment_embeddings/"
        ),
        help="Output dataset dir",
    )

    args = parser.parse_args()

    embed_autosegmented(
        pose_path=args.pose_path,
        autoseg_model=args.autoseg_model,
        embedding_model=args.embedding_model,
        segments_json=args.segments_json,
        output_dir=args.output_dir,
    )
# find "/opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/webdataset_extracted/ase/chronological_bible_translation_in_american_sign_language_119_introductions_and_passages/" -name "*.pose"| parallel -j2 python embed_autosegmented_windows.py "{}" --output_dir /data/petabyte/cleong/data/DBL_Deaf_Bibles/autosegment_embeddings/


# find /data/petabyte/cleong/data/DBL_Deaf_Bibles/webdataset_extracted/ \
#   -mindepth 2 -maxdepth 2 -type d \
#   | parallel --progress -j6 '
#       rel={= s:.*/webdataset_extracted/:: =}
#       out="/data/petabyte/cleong/data/DBL_Deaf_Bibles/autosegment_embeddings/$rel/"
#       mkdir -p "$out"
#       find "{}" -name "*.pose" \
#         | parallel -j2 python embed_autosegmented_windows.py "{}" --output_dir "$out"
#     '


# (signclip_inference) root@16-cpu-128g-1gshared-exclgpu:/opt/home/cleong/projects/semantic-sign-language-search/setup_signCLIP/fairseq/examples/MMPT# find "/data/petabyte/cleong/data/DBL_Deaf_Bibles/webdataset_extracted/" -type f -name "*.pose"|parallel --progress -j 8 python embed_autosegmented_windows.py "{}"
