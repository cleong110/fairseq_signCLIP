from collections import defaultdict
import json
import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from pose_format import Pose
from mmpt.models import MMPTModel
import matplotlib.pyplot as plt
import time
import logging

from sign_language_gloss_utils.glosses.text_utils import (
    get_glosses_set_from_text,
    preprocess_text as lemmatize,
)
from contextlib import contextmanager

import os
from tqdm import tqdm
from functools import cache


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow C++ backend logs
os.environ["GLOG_minloglevel"] = "3"  # Suppress GLOG messages from XLA/CUDA

import warnings

warnings.filterwarnings("ignore")  # Suppress Python warnings


logging.getLogger("tensorflow").setLevel(
    logging.ERROR
)  # Suppress TensorFlow Python logs


# Optionally, if using Hugging Face Transformers, you can suppress its logs too:
try:
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()
except ImportError:
    pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,  # <— this overwrites any prior logging setup
)
logger = logging.getLogger(__name__)


@contextmanager
def timed_section(name: str):
    start = time.perf_counter()
    result = {}
    yield result
    end = time.perf_counter()
    duration = end - start
    result["duration"] = duration
    logger.info(f"[TIMER] {name} took {duration:.2f} seconds.")


SignCLIPEmbeddable = Pose | str | Path

# import mediapipe as mp
# mp_holistic = mp.solutions.holistic
# FACEMESH_CONTOURS_POINTS = [str(p) for p in sorted(set(p for p_tup in mp_holistic.FACEMESH_CONTOURS for p in p_tup))]

# To avoid installing mediapipe, we just hardcode the face contours given the above code
FACEMESH_CONTOURS_POINTS = [
    "0",
    "7",
    "10",
    "13",
    "14",
    "17",
    "21",
    "33",
    "37",
    "39",
    "40",
    "46",
    "52",
    "53",
    "54",
    "55",
    "58",
    "61",
    "63",
    "65",
    "66",
    "67",
    "70",
    "78",
    "80",
    "81",
    "82",
    "84",
    "87",
    "88",
    "91",
    "93",
    "95",
    "103",
    "105",
    "107",
    "109",
    "127",
    "132",
    "133",
    "136",
    "144",
    "145",
    "146",
    "148",
    "149",
    "150",
    "152",
    "153",
    "154",
    "155",
    "157",
    "158",
    "159",
    "160",
    "161",
    "162",
    "163",
    "172",
    "173",
    "176",
    "178",
    "181",
    "185",
    "191",
    "234",
    "246",
    "249",
    "251",
    "263",
    "267",
    "269",
    "270",
    "276",
    "282",
    "283",
    "284",
    "285",
    "288",
    "291",
    "293",
    "295",
    "296",
    "297",
    "300",
    "308",
    "310",
    "311",
    "312",
    "314",
    "317",
    "318",
    "321",
    "323",
    "324",
    "332",
    "334",
    "336",
    "338",
    "356",
    "361",
    "362",
    "365",
    "373",
    "374",
    "375",
    "377",
    "378",
    "379",
    "380",
    "381",
    "382",
    "384",
    "385",
    "386",
    "387",
    "388",
    "389",
    "390",
    "397",
    "398",
    "400",
    "402",
    "405",
    "409",
    "415",
    "454",
    "466",
]

MAX_FRAMES_DEFAULT = 256  # Default truncate length, can be overridden

# Model configurations (keep these unchanged)
model_configs = [
    # ("default", "signclip_v1_1/baseline_temporal"), # multilingual pretrained
    ("default", "semantic-search/embed_with_asl_finetune_checkpoint_best"),
    (
        "asl_finetune_checkpoint_best",
        "semantic-search/embed_with_asl_finetune_checkpoint_best",
    ),
    # size mismatch for video_encoder.bert.embeddings.word_embeddings.weight: copying a param with shape torch.Size([28996, 768]) from checkpoint, the shape in current model is torch.Size([30522, 768]).
    # runs/signclip_embed/asl_citizen_finetune_checkpoint_best/checkpoint_best.pt:       Key: mmmodel.video_encoder.bert.embeddings.word_embeddings.weight (Tensor) shape: (28996, 768) dtype: torch.float32
    # runs/signclip_embed/asl_citizen_finetune_checkpoint_best/checkpoint_best.pt:       Key: mmmodel.text_encoder.embeddings.word_embeddings.weight (Tensor) shape: (28996, 768) dtype: torch.float32
    # https://huggingface.co/google-bert/bert-base-uncased?show_file_info=model.safetensors: bert.embeddings.word_embeddings.weight 	[30 522, 768]
    # https://huggingface.co/google-bert/bert-base-multilingual-cased?show_file_info=model.safetensors: bert.embeddings.word_embeddings.weight 	[119 547, 768]
    # ("asl_finetune_checkpoint_best_uncased",
    #     "semantic-search/embed_with_asl_finetune_checkpoint_best_and_bert_uncased"),
    ("baseline_temporal", "semantic-search/embed_with_baseline_temporal"),
    # ("asl_citizen", "signclip_asl/asl_citizen_finetune"), # fine-tuned on ASL Citizen
    # ("asl_finetune", "signclip_asl/asl_finetune"), # fine-tuned on three ASL datasets
    # ("suisse", "signclip_suisse/suisse_finetune"), # fine-tuned on Signsuisse
    # below are config files for longer duration inference and absuluate checkpoint path
    # ("default", "signclip_v1_1/baseline_temporal_inference"), # multilingual pretrained
    # ("suisse", "signclip_suisse/suisse_finetune_inference"), # fine-tuned on Signsuisse
]

# Cache for models that have been lazily initialized.
models_cache = {}


def get_model(model_name):
    """
    Lazily load the requested model based on model_name.
    If the model is already loaded, return it.
    Otherwise, find its config, load it, and cache it.
    """
    if model_name in models_cache:
        return models_cache[model_name]

    # Look up the configuration for the given model_name.
    config_path = None
    for m_name, cfg in model_configs:
        if m_name == model_name:
            config_path = cfg
            break

    if config_path is None:
        raise ValueError(f"Unknown model name: {model_name}")
    file_dir = Path(__file__).parent.resolve()
    # Load the model, tokenizer, and aligner.
    model, tokenizer, aligner = MMPTModel.from_pretrained(
        file_dir / f"projects/retri/{config_path}.yaml",
        # f"/home/zifjia/fairseq/examples/MMPT/projects/retri/{config_path}.yaml",
        video_encoder=None,
    )
    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    models_cache[model_name] = {
        "model": model,
        "tokenizer": tokenizer,
        "aligner": aligner,
    }
    return models_cache[model_name]


def pose_normalization_info(pose_header):
    if pose_header.components[0].name == "POSE_LANDMARKS":
        return pose_header.normalization_info(
            p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
            p2=("POSE_LANDMARKS", "LEFT_SHOULDER"),
        )

    if pose_header.components[0].name == "BODY_135":
        return pose_header.normalization_info(
            p1=("BODY_135", "RShoulder"), p2=("BODY_135", "LShoulder")
        )

    if pose_header.components[0].name == "pose_keypoints_2d":
        return pose_header.normalization_info(
            p1=("pose_keypoints_2d", "RShoulder"), p2=("pose_keypoints_2d", "LShoulder")
        )

    raise ValueError(
        f"Could not parse normalization info, pose_header.components[0].name is {pose_header.components[0].name}. Expected one of (POSE_LANDMARKS,BODY_135,pose_keypoints_2d)"
    )


def pose_hide_legs(pose):
    if pose.header.components[0].name == "POSE_LANDMARKS":
        point_names = ["KNEE", "ANKLE", "HEEL", "FOOT_INDEX"]
        # pylint: disable=protected-access
        points = [
            pose.header._get_point_index("POSE_LANDMARKS", side + "_" + n)
            for n in point_names
            for side in ["LEFT", "RIGHT"]
        ]
        pose.body.confidence[:, :, points] = 0
        pose.body.data[:, :, points, :] = 0
        return pose
    raise ValueError("Unknown pose header schema for hiding legs")


def preprocess_pose(pose, max_frames=None):
    pose = pose.get_components(
        [
            "POSE_LANDMARKS",
            "FACE_LANDMARKS",
            "LEFT_HAND_LANDMARKS",
            "RIGHT_HAND_LANDMARKS",
        ],
        {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS},
    )

    pose = pose.normalize(pose_normalization_info(pose.header))
    pose = pose_hide_legs(pose)

    feat = np.nan_to_num(pose.body.data)
    feat = feat.reshape(feat.shape[0], -1)

    pose_frames = torch.from_numpy(
        np.expand_dims(feat, axis=0)
    ).float()  # e.g., torch.Size([1, frame count, 609])
    if max_frames is not None and pose_frames.size(1) > max_frames:
        logger.warning(
            f"pose sequence length too long ({pose_frames.size(1)}) longer than {max_frames} frames. Truncating"
        )
        pose_frames = pose_frames[:, :max_frames, :]

    return pose_frames


def preprocess_text(text, model_name="default"):
    model_info = get_model(model_name)
    aligner = model_info["aligner"]
    tokenizer = model_info["tokenizer"]

    caps, cmasks = aligner._build_text_seq(
        tokenizer(text, add_special_tokens=False)["input_ids"],
    )
    caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1

    return caps, cmasks


def embed_pose(pose, model_name="default", max_frames=None):
    model_info = get_model(model_name)
    model = model_info["model"]

    caps, cmasks = preprocess_text("", model_name)
    poses = pose if isinstance(pose, list) else [pose]
    embeddings = []

    pose_frames_l = []
    for p in poses:
        pose_frames = preprocess_pose(p, max_frames=max_frames)
        pose_frames_l.append(pose_frames)

    # Batch padding
    # 1) find the longest sequence
    max_len = max(pf.shape[1] for pf in pose_frames_l)

    # 2) pad each to max_len on the time‐axis
    padded = []
    for pf in pose_frames_l:
        pad_len = max_len - pf.shape[1]
        if pad_len > 0:
            # create zeros of shape (batch=1, pad_len, features)
            pad = pf.new_zeros((pf.shape[0], pad_len, pf.shape[2]))
            pf = torch.cat([pf, pad], dim=1)
        padded.append(pf)

    # 3) now you can concatenate without size mismatches
    pose_frames_l = torch.cat(padded, dim=0)

    batch_size = len(poses)

    with torch.no_grad():
        output = model(
            pose_frames_l,
            caps.repeat(batch_size, 1),
            cmasks.repeat(batch_size, 1),
            return_score=False,
        )
        embeddings.append(output["pooled_video"].cpu().numpy())

    return np.concatenate(embeddings)


@cache
def embed_text(text, model_name="default"):
    model_info = get_model(model_name)
    model = model_info["model"]

    # Determine the placeholder dimension based on the model_name.
    if model_name == "lip":
        placeholder_dim = 1377
    elif model_name == "lip_only":
        placeholder_dim = 768
    else:
        placeholder_dim = 609

    # Ensure texts is a list.
    texts = text if isinstance(text, list) else [text]
    batch_size = len(texts)

    # Preprocess each text individually and store the results.
    caps_list = []
    cmasks_list = []
    for t in texts:
        caps, cmasks = preprocess_text(t, model_name)
        caps_list.append(caps)  # Each should have shape (1, 128)
        cmasks_list.append(cmasks)

    # Concatenate the individual results along the batch dimension.
    caps_batch = torch.cat(caps_list, dim=0)
    cmasks_batch = torch.cat(cmasks_list, dim=0)

    # Create dummy pose_frames with shape (batch_size, 1, placeholder_dim).
    pose_frames = torch.randn(batch_size, 1, placeholder_dim)

    # Run the model forward pass only once with the full batch.
    with torch.no_grad():
        output = model(pose_frames, caps_batch, cmasks_batch, return_score=False)

    # Extract the pooled text embeddings and return as a NumPy array.
    embeddings = output["pooled_text"].cpu().numpy()
    return embeddings


def score_pose_and_text(pose, text, model_name="default", max_frames=None):
    model_info = get_model(model_name)
    model = model_info["model"]

    pose_frames = preprocess_pose(pose, max_frames)
    caps, cmasks = preprocess_text(text, model_name)

    with torch.no_grad():
        output = model(pose_frames, caps, cmasks, return_score=True)

    return text, float(output["score"])  # dot-product


def score_pose_and_text_batch(pose, text, model_name="default"):
    pose_embedding = embed_pose(pose, model_name)
    text_embedding = embed_text(text, model_name)

    scores = np.matmul(pose_embedding, text_embedding.T)
    return scores


@cache
def embed_either_format(input_item: SignCLIPEmbeddable, max_frames=None) -> np.ndarray:
    if isinstance(input_item, Path):
        return embed_pose(Pose.read(input_item.read_bytes()), max_frames=max_frames)
    if isinstance(input_item, str):
        return embed_text(f"<en> <ase> {input_item}")
    elif isinstance(input_item, Pose):
        return embed_pose(input_item, max_frames=max_frames)


def score_either_format(query: SignCLIPEmbeddable, target: SignCLIPEmbeddable):
    query_embed = embed_either_format(query, max_frames=MAX_FRAMES_DEFAULT)
    second_embed = embed_either_format(target)
    score = np.matmul(query_embed, second_embed.T).squeeze()
    return float(score)


def load_pose(
    pose_path: Path, start_time_ms: int | None = None, end_time_ms: int | None = None
) -> Pose:
    """
    Loads a pose file and optionally slices it by time.

    Parameters:
        pose_path: Path to the `.pose` file.
        start_time_ms: Optional start time in milliseconds.
        end_time_ms: Optional end time in milliseconds.

    Returns:
        A Pose object sliced by the specified time range.
    """
    if not pose_path.is_file():
        raise FileNotFoundError(f"Pose file not found: {pose_path}")

    with open(pose_path, "rb") as f:
        return Pose.read(f, start_time=start_time_ms, end_time=end_time_ms)


@cache
def load_window_embedding(
    pose_path: Path,
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
) -> np.ndarray:
    """
    Load embeddings for a pose window. If a cached parquet file of embeddings exists,
    load from there. Otherwise, compute embeddings, save them for future use, and return.

    Multiple windows per pose file are supported; the parquet grows as new windows are added.
    """
    parquet_path = pose_path.with_suffix(".window_embeddings.parquet")

    if parquet_path.exists():
        df = pd.read_parquet(
            parquet_path, columns=["start_time_ms", "end_time_ms", "embedding"]
        )

        # Try to find matching row for this window
        mask = (df["start_time_ms"] == start_time_ms) & (
            df["end_time_ms"] == end_time_ms
        )
        if mask.any():
            logger.info("Loading cached embeddings from %s", parquet_path)
            row = df.loc[mask].iloc[0]
            return np.array(row["embedding"], dtype=np.float32).reshape(1, -1)

    # Fall back to computing embeddings
    logger.info(
        "No cached embeddings found. Computing embeddings for %s [%s - %s]",
        pose_path,
        start_time_ms,
        end_time_ms,
    )
    poses = load_pose(
        pose_path=pose_path, start_time_ms=start_time_ms, end_time_ms=end_time_ms
    )
    embeddings = embed_pose(poses)  # shape (1, 768)

    # Flatten to 1D list for storage
    row = {
        "start_time_ms": start_time_ms,
        "end_time_ms": end_time_ms,
        "embedding": embeddings.flatten().tolist(),
    }

    # Append to parquet (read, concat, write back)
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    logger.debug(f"Saving updated embeddings {embeddings.shape} to %s", parquet_path)
    df.to_parquet(parquet_path, index=False)

    return embeddings


def get_sliding_window_frame_ranges(
    max: int,
    window_length: int = 1000,
    stride: int = 500,
    start: int = 0,
    end: int | None = None,
) -> list[tuple[int, int]]:
    end = end if end is not None else max

    if end > max:
        raise ValueError(f"End index {end} exceeds max length {max}")
    if window_length <= 0 or stride <= 0:
        raise ValueError("window_length and stride must be > 0")

    ranges = []
    i = start
    while i < end:
        window_end = min(i + window_length, end)
        ranges.append((i, window_end))
        if window_end == end:
            break
        i += stride
    return ranges


def score_windows_over_pose(
    pose_path,
    query: SignCLIPEmbeddable,
    start_ms,
    end_ms,
    window_size_ms,
    step_size_ms,
    model_name,
):
    """
    Slide a window over the pose file and score each window using a precomputed query embedding.

    Returns:
        List of tuples: (window_start_ms, window_end_ms, score)
    """
    query_scores = defaultdict(list)
    query_embed = embed_either_format(query, max_frames=MAX_FRAMES_DEFAULT)

    for window_start, window_end in tqdm(
        get_sliding_window_frame_ranges(
            max=end_ms,
            window_length=window_size_ms,
            stride=step_size_ms,
            start=start_ms,
            end=end_ms,
        ),
        desc="sliding over windows",
    ):
        # window_end = min(window_start + window_size_ms, end_ms)
        # with timed_section(f"Load Target Pose Window")
        # pose = load_pose(
        #     pose_path=pose_path, start_time_ms=window_start, end_time_ms=window_end
        # )
        # pose = load_pose(
        #     pose_path=pose_path, start_time_ms=window_start, end_time_ms=window_end
        # )
        pose_embedding = load_window_embedding(
            pose_path=pose_path, start_time_ms=window_start, end_time_ms=window_end
        )

        score = float(np.matmul(query_embed, pose_embedding.T).squeeze())

        query_scores["window_start_ms"].append(window_start)
        query_scores["window_midpoint_ms"].append((window_start + window_end) // 2)
        query_scores["window_end_ms"].append(window_end)
        query_scores["score"].append(score)
        # query_scores.append((window_start, window_end, score))

    return pd.DataFrame(query_scores)


def get_pose_queries(query_glosses, dataset_df, samples_per_gloss=5):
    gloss_df = dataset_df[dataset_df["GLOSS"].isin(query_glosses)].copy()
    # Group by gloss and sample
    # TODO: check if the pose is longer than 256 frames
    sampled_df = gloss_df.groupby("GLOSS", group_keys=False).apply(
        lambda x: x.sample(min(len(x), samples_per_gloss), random_state=42)
    )
    for row in sampled_df[["GLOSS", "POSE_FILE_PATH"]].itertuples(
        index=False, name=None
    ):
        gloss, pose_path = row
        query_label = gloss
        query_id = Path(pose_path).stem
        query_value = Path(pose_path)
        yield query_label, query_id, query_value


def get_text_queries(query_glosses):
    text_queries = []
    for gloss in query_glosses:
        for transform in ["upper", "lower", "capitalize"]:
            query_label = gloss

            if transform == "capitalize":
                query_value = gloss.capitalize()
            elif transform == "lower":
                query_value = gloss.lower()
            else:
                query_value = gloss.upper()

            query_id = f"{query_value} (eng)"

            text_queries.append((query_label, query_id, query_value))

    return text_queries


def score_with_text_and_gloss_keywords(
    text_query,
    pose_path,
    pose_dataset_csv,
    start_time_ms,
    end_time_ms,
    window_size_ms,
    step_size_ms,
    model_to_use,
    results_dir,
    eng_text_label="",
    samples_per_gloss=5,
    include_lemmas=True,
):
    if not pose_path.is_file():
        raise FileNotFoundError(f"Pose file not found: {pose_path}")

    dataset_df = pd.read_csv(pose_dataset_csv)
    dataset_vocab = set(dataset_df["GLOSS"].unique())
    logger.info(f"Loaded dataset with vocab of length: {len(dataset_vocab)}")

    if not include_lemmas:
        query_glosses = get_glosses_set_from_text(
            text_query, dataset_vocab, remove_stopwords=True
        )
    else:
        query_glosses = set(lemmatize(text_query))

    logger.info(f"Glosses found for '{text_query}': {query_glosses}")

    pose_queries = list(
        get_pose_queries(query_glosses, dataset_df, samples_per_gloss=samples_per_gloss)
    )
    text_queries = get_text_queries(query_glosses)

    queries = (
        text_queries + pose_queries
    )  # + [("full_text_query", text_query[:100],text_query)]

    logger.info(f"Total Queries: {len(queries)}")

    for q in queries:
        logger.info(f"Gloss/Keyword Query: {q}")

    full_pose = load_pose(pose_path)
    duration_ms = 1000 * len(full_pose.body.data) / full_pose.body.fps

    # run_dir = results_dir / model_to_use / pose_path.stem / str(uuid.uuid4())

    queries_dir = results_dir / "queries"
    queries_dir.mkdir(parents=True, exist_ok=True)

    all_query_score_dfs = []

    for query_label, query_id, query_value in queries:
        query_start = start_time_ms if start_time_ms is not None else 0
        query_end = end_time_ms if end_time_ms is not None else duration_ms - 1

        logger.debug(
            f"Running query '{query_id}' with label {query_label} from {query_start}ms to {query_end:.2f}ms"
        )

        query_dir = queries_dir / query_id
        query_dir.mkdir(exist_ok=True, parents=True)

        out_png = (
            query_dir
            / f"{query_id}_scores_{query_start}_to_{query_end}_step{window_size_ms}.png"
        )

        query_scores_df = score_windows_over_pose(
            pose_path=pose_path,
            query=query_value,
            start_ms=query_start,
            end_ms=query_end,
            window_size_ms=window_size_ms,
            step_size_ms=step_size_ms,
            model_name=model_to_use,
        )

        query_scores_df["eng"] = eng_text_label
        query_scores_df["query_label"] = query_label
        query_scores_df["query_id"] = query_id

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(
            query_scores_df["window_midpoint_ms"], query_scores_df["score"], marker="o"
        )
        plt.title(
            f"Score of '{query_id}' over Sliding Windows\n(step={step_size_ms}ms, window={window_size_ms}ms)"
        )
        plt.xlabel("Window Midpoint Time (ms)")
        plt.ylabel("Score")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_png)
        logger.info(f"Saved plot: {out_png.resolve()}")

        # Save per-query scores
        scores_out = query_dir / "scores.parquet"
        query_scores_df.to_parquet(scores_out)
        logger.info(f"Saved scores: {scores_out.resolve()}")

        all_query_score_dfs.append(query_scores_df)

    # TODO: what if these are empty
    if len(all_query_score_dfs) > 0:
        all_query_scores_df = pd.concat(all_query_score_dfs)
    else:
        all_query_scores_df = pd.DataFrame()

    all_query_scores_df["text_query"] = text_query
    all_query_scores_df["target_video_name"] = pose_path.name
    all_query_scores_df["samples_per_gloss"] = samples_per_gloss
    all_query_scores_df["search_start_time_ms"] = start_time_ms
    all_query_scores_df["search_end_time_ms"] = end_time_ms
    all_query_scores_df["search_window_size_ms"] = window_size_ms
    all_query_scores_df["search_step_size_ms"] = step_size_ms
    all_query_scores_df["model_to_use"] = model_to_use
    logger.info(f"Finished all queries for {pose_path.name}")
    return all_query_scores_df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pose and text similarity using SignCLIP."
    )
    parser.add_argument(
        "--pose_path",
        default="./house.pose",
        type=Path,
        help="Path to the .pose file.",
    )
    parser.add_argument(
        "--max_frames",
        nargs="?",
        type=int,
        const=MAX_FRAMES_DEFAULT,
        default=None,
        help=f"If provided, pose sequences longer than this will be truncated, otherwise they will not. If provided without a value, will use {MAX_FRAMES_DEFAULT}, as SignCLIP can currently only support this many. If provided with a value, will use that value",
    )
    parser.add_argument(
        "--start_time_ms",
        type=int,
        default=None,
        help="Optional start time (in ms) to slice the pose.",
    )
    parser.add_argument(
        "--end_time_ms",
        type=int,
        default=None,
        help="Optional end time (in ms) to slice the pose.",
    )
    # God called the light day and the darkness He called night
    parser.add_argument(
        "--window_size_ms",
        type=int,
        default=1000,
        help="window length for pose slicing",
    )
    parser.add_argument(
        "--step_size_ms",
        type=int,
        default=500,
        help="Steps for slicing the pose",
    )

    parser.add_argument(
        "--eng",
        type=str,
        # default="god",
        help="English text to look for (comma-separated)",
    )

    parser.add_argument(
        "--prominence",
        type=float,
        # default="god",
        help="Passed to the find_peaks function for finding 'hits'",
    )

    parser.add_argument(
        "--height_multiplier",
        type=float,
        help="Passed to the find_peaks function for finding 'hits'",
    )

    parser.add_argument(
        "--density_weight",
        type=float,
        help="How much to weight hit density. ",
    )

    parser.add_argument(
        "--samples-per-gloss",
        type=int,
        default=5,
        help="Samples per gloss",
    )

    parser.add_argument(
        "--query-json",
        type=Path,
        help="JSON with queries in it. Otherwise will look for one at pose_path.transcripts.json",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="default",
        # choices=[str(key) for key, _ in model_configs],
        help="Model to use (comma-separated)",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Where to save results",
    )

    parser.add_argument(
        "--pose-dataset-csv",
        type=Path,
        default=Path(
            "/opt/home/cleong/projects/pose-evaluation/dataset_dfs/asl-citizen.csv"
        ),
        help="Path to a CSV file with columns GLOSS, POSE_FILE_PATH (default: %(default)s).",
    )

    args = parser.parse_args()

    segment_indices = []
    text_queries = []

    run_dir = (
        args.results_dir
        / args.model
        / f"samplespergloss_{args.samples_per_gloss}"
        / f"start_{args.start_time_ms}_end_{args.end_time_ms}"
        / f"windowsize{args.window_size_ms}_step{args.step_size_ms}"
        / args.pose_path.stem
        # / f"{uuid.uuid4()}"
    )

    logger.info(f"Run Dir: {run_dir}")

    if args.eng is not None:
        for eng_query in args.eng.split(","):
            text_queries.append(eng_query)
            segment_indices.append(9999)
            logger.info(f"Text Queries from CLI: {text_queries}")

    else:
        text_queries.append("")
        segment_indices.append(-1)

    # --------------- Iterate over segments
    for seg_idx, text_query in zip(segment_indices, text_queries):
        # print(f"Query for segment #{seg_idx}:\n\tq: {text_query}\n")
        if seg_idx < 9999 or text_query == "":
            continue
        segment_dir = run_dir / f"seg_idx{seg_idx}"
        segment_dir.mkdir(exist_ok=True, parents=True)
        logger.info(
            f"Query for segment #{seg_idx}:\n\tq: {text_query}\n\tout: {segment_dir}"
        )

        if not text_query:
            continue
        # ----------- Get "Keyword" scores
        with timed_section("Get Keyword Scores") as t:
            df = score_with_text_and_gloss_keywords(
                text_query=text_query,
                pose_path=args.pose_path,
                pose_dataset_csv=args.pose_dataset_csv,
                start_time_ms=args.start_time_ms,
                end_time_ms=args.end_time_ms,
                window_size_ms=args.window_size_ms,
                step_size_ms=args.step_size_ms,
                model_to_use=args.model,
                results_dir=segment_dir,
                eng_text_label=args.eng,
                samples_per_gloss=args.samples_per_gloss,
            )

        run_dir.mkdir(parents=True, exist_ok=True)

        pose = Pose.read(args.pose_path.read_bytes())

        pose_info = {
            "fps": pose.body.fps,
            "frame_count": pose.body.data.shape[0],
            "people": pose.body.data.shape[1],
            "keypoints": pose.body.data.shape[2],
            "path": str(args.pose_path),
        }
        with open(run_dir / "pose_info.json", "w") as f:
            json.dump(pose_info, f)

        df["query_group_score_time"] = t
        df["query_seg_idx"] = seg_idx
        df.to_parquet(segment_dir / "all_scores.parquet")
        logger.info(f"Saved all scores to {segment_dir / 'all_scores.parquet'}")


if __name__ == "__main__":
    main()


# Conda env:
# conda activate /opt/home/cleong/envs/signclip_inference/
# python /opt/home/cleong/projects/semantic-sign-language-search/setup_signCLIP/fairseq/examples/MMPT/search_with_text_and_poses.py --pose_path "test_data/ase_chronological_bible_translation_in_american_sign_language_119_introductions_and_passages_cbt-001-ase-3-passage _ god creates the world.pose-mediapipe.pose" --start_time_ms 5000 --end_time_ms 60000 --step_size_ms 10 --window_size_ms 1500 --eng "god life heaven sky earth" --model "asl_finetune_checkpoint_best"


# find test_data/ -name "*.pose"|parallel -j2 python /opt/home/cleong/projects/semantic-sign-language-search/setup_signCLIP/fairseq/examples/MMPT/search_with_text_and_poses.py --pose_path "{}" --start_time_ms 0 --step_size_ms 10 --window_size_ms 1000 --model "asl_finetune_checkpoint_best"
# find test_data/ -name "*.pose"|parallel -j2 python /opt/home/cleong/projects/semantic-sign-language-search/setup_signCLIP/fairseq/examples/MMPT/search_with_text_and_poses.py --pose_path "{}" --start_time_ms 0 --step_size_ms 100 --window_size_ms 3000 --eng "god" --model "asl_finetune_checkpoint_best"

# just God creates the world, and also search for "refrigerator"
# find test_data/ -name "*passage _ god creates the world.pose-mediapipe*.pose"|parallel -j1 python /opt/home/cleong/projects/semantic-sign-language-search/setup_signCLIP/fairseq/examples/MMPT/search_with_text_and_poses.py --pose_path "{}" --start_time_ms 0 --step_size_ms 100 --window_size_ms 1000 --eng "refrigerator" --model "asl_finetune_checkpoint_best"


# add in 15 queries/gloss and search God Creates the World, and search with step 200
# find test_data/ -name "*passage _ god creates the world.pose-mediapipe*.pose"|parallel -j1 python /opt/home/cleong/projects/semantic-sign-language-search/setup_signCLIP/fairseq/examples/MMPT/search_with_text_and_poses.py --pose_path "{}" --start_time_ms 0 --step_size_ms 200 --window_size_ms 1000 --eng "refrigerator,god,heaven,day,earth" --model "asl_finetune_checkpoint_best" --samples-per-gloss 15

# Same, but just The First Man and Womand Disobey God
# find /data/petabyte/cleong/data/DBL_Deaf_Bibles/webdataset_extracted/ -name "*passage _ *first*man*and*woman*disobey*.pose-mediapipe*.pose"|parallel -j1 python /opt/home/cleong/projects/semantic-sign-language-search/setup_signCLIP/fairseq/examples/MMPT/search_with_text_and_poses.py --pose_path "{}" --start_time_ms 0 --step_size_ms 200 --window_size_ms 1000 --eng "refrigerator" --model "asl_finetune_checkpoint_best" --samples-per-gloss 15


# (signclip_inference) root@16-cpu-128g-1gshared-exclgpu:/opt/home/cleong/projects/semantic-sign-language-search/setup_signCLIP/fairseq/examples/MMPT# python search_with_text_and_poses.py --eng "SILVER ISRAEL MAN FAMILY SON COME PEOPLE SELECT TENT ENEMY HIDE BURN TROUBLE BAR BRING" --samples-per-gloss 16 --pose_path "/data/petabyte/cleong/data/DBL_Deaf_Bibles/webdataset_extracted/ase/chronological_bible_translation_in_american_sign_language_119_introductions_and_passages/ase_chronological_bible_translation_in_american_sign_language_119_introductions_and_passages_cbt-033-ase-2-passage _ israel fails to conquer ai.pose-mediapipe.pose" --start_time_ms 0 --step_size_ms 200 --window_size_ms 1000 --model "asl_finetune_checkpoint_best"

# python search_with_text_and_poses.py --eng "YAHWEH GOD" --samples-per-gloss 26 --pose_path "/data/petabyte/cleong/data/DBL_Deaf_Bibles/webdataset_extracted/ase/chronological_bible_translation_in_american_sign_language_119_introductions_and_passages/ase_chronological_bible_translation_in_american_sign_language_119_introductions_and_passages_cbt-001-ase-3-passage _ god creates the world.pose-mediapipe.pose" --start_time_ms 0 --step_size_ms 100 --window_size_ms 500 --model "asl_finetune_checkpoint_best"
