import argparse
import json
import logging
from pathlib import Path

from pose_format import Pose

import pandas as pd
import pyarrow.dataset as ds
from tqdm import tqdm

from search_with_text_and_poses import embed_pose, embed_text, MAX_FRAMES_DEFAULT

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_pose(
    pose_path: Path, start_frame: int | None = None, end_frame: int | None = None
) -> Pose:
    """
    Loads a pose file and optionally slices it by frame.

    Parameters:
        pose_path: Path to the `.pose` file.
        start_time_ms: Optional start frame
        end_time_ms: Optional end frame

    Returns:
        A Pose object sliced by the specified time range.
    """
    if not pose_path.is_file():
        raise FileNotFoundError(f"Pose file not found: {pose_path}")

    with open(pose_path, "rb") as f:
        return Pose.read(f, start_frame=start_frame, end_frame=end_frame)


def load_transcript_segments(transcript_path: Path) -> list[dict]:
    """
    Load transcript JSON segments.
    Each segment is expected to have:
        - text
        - start_frame
        - end_frame
        - language info
        - license, source
        - bible-ref
    """
    with open(transcript_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    logger.info(f"Loaded {len(segments)} segments from {transcript_path.name}")
    return segments


def get_existing_segments(
    dataset_dir: Path, model: str, video_stem: str, columns=None
) -> pd.DataFrame:
    """
    Return DataFrame of already-embedded transcript segments for a given model/video.
    Ensures consistent schema even if directory does not exist or is empty.
    """
    if not dataset_dir.exists() or not any(dataset_dir.rglob("*.parquet")):
        # return empty dataframe with consistent columns
        return pd.DataFrame(
            columns=[
                "video_stem",
                "segment_index",
                "start_frame",
                "end_frame",
                "text",
                "bible_ref",
                "embedding",
                "embedding_model",
            ]
        )

    dataset = ds.dataset(dataset_dir, format="parquet", partitioning="hive")
    filt = (ds.field("embedding_model") == model) & (
        ds.field("video_stem") == video_stem
    )
    table = dataset.to_table(filter=filt, columns=columns)
    return table.to_pandas()


def embed_transcript(
    transcript_path: Path,
    pose_path: Path,
    embedding_model: str,
    output_dir: Path,
    truncate=True,
) -> None:
    """
    Embed transcript segments for a single video and save parquet dataset with hive partitioning.
    """
    video_stem = pose_path.name.split(".")[
        0
    ]  # strip all extensions to get consistent stem

    if transcript_path is None:
        transcript_path = pose_path.parent / f"{video_stem}.transcripts.json"
        logger.info(f"Derived transcript path from pose path: {transcript_path}")
    segments = load_transcript_segments(transcript_path)

    existing = get_existing_segments(output_dir, embedding_model, video_stem)
    existing_set = set(
        zip(existing["segment_index"], existing["start_frame"], existing["end_frame"])
    )

    rows = []
    for idx, seg in enumerate(tqdm(segments, desc=f"Embedding {video_stem}")):
        key = (idx, seg["start_frame"], seg["end_frame"])
        if key in existing_set:
            continue

        # load pose segment using frame indices
        seg_length = seg["end_frame"] - seg["start_frame"]
        if seg_length >= MAX_FRAMES_DEFAULT:
            end_frame_to_embed = seg["start_frame"] + 256
        else:
            end_frame_to_embed = seg["start_frame"]

        pose = load_pose(
            pose_path=pose_path,
            start_frame=seg["start_frame"],
            end_frame=end_frame_to_embed,
        )

        try:
            pose_embedding = embed_pose(pose, model_name=embedding_model).squeeze()
            text = seg.get("text", "")
            text_embedding = embed_text(
                f"<en> <ase> {text}", model_name=embedding_model
            ).squeeze()
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
                "segment_index": idx,
                "start_frame": seg["start_frame"],
                "end_frame": seg["end_frame"],
                "end_frame_embedded": end_frame_to_embed,
                "text": seg.get("text", ""),
                "text_embedding": text_embedding.tolist(),
                "bible_ref": seg.get("bible-ref"),
                "embedding": pose_embedding.tolist(),
                "embedding_model": embedding_model,
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
        partition_cols=["embedding_model", "video_stem"],
    )

    logger.info(
        f"Saved {len(df)} new transcript embeddings for {video_stem} to {output_dir.resolve()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed transcript segments for a video"
    )
    parser.add_argument(
        "pose_path", type=Path, help="Path to .pose-mediapipe.pose file"
    )
    parser.add_argument("--transcript_path", type=Path, help="Path to transcript JSON")
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="asl_finetune_checkpoint_best",
        help="Embedding model name",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("embeddings_transcripts"),
        help="Output dataset directory",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    embed_transcript(
        transcript_path=args.transcript_path,
        pose_path=args.pose_path,
        embedding_model=args.embedding_model,
        output_dir=args.output_dir,
    )


# python embed_transcript_windows.py "/opt/home/cleong/projects/semantic_and_visual_similarity/nas_data/DBL_Deaf_Bibles/webdataset_extracted/ase/chronological_bible_translation_in_american_sign_language_119_introductions_and_passages/ase_chronological_bible_translation_in_american_sign_language_119_introductions_and_passages_cbt-001-ase-3-passage _ god creates the world.pose-mediapipe.pose" --output_dir /data/petabyte/cleong/data/DBL_Deaf_Bibles/transcript_embeddings/

# find "/opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/webdataset_extracted/ase/chronological_bible_translation_in_american_sign_language_119_introductions_and_passages/" -name "*.pose"| parallel -j2 python embed_transcript_windows.py "{}" --output_dir /data/petabyte/cleong/data/DBL_Deaf_Bibles/transcript_embeddings/
