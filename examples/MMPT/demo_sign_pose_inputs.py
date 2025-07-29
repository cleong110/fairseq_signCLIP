from collections import defaultdict
import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from pose_format import Pose
from mmpt.models import MMPTModel
import matplotlib.pyplot as plt

import os
import itertools
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow C++ backend logs
os.environ["GLOG_minloglevel"] = "3"  # Suppress GLOG messages from XLA/CUDA

# import warnings
# warnings.filterwarnings("ignore")           # Suppress Python warnings

import logging

logging.getLogger("tensorflow").setLevel(
    logging.ERROR
)  # Suppress TensorFlow Python logs

# Optionally, if using Hugging Face Transformers, you can suppress its logs too:
try:
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()
except ImportError:
    pass


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

    # Load the model, tokenizer, and aligner.
    model, tokenizer, aligner = MMPTModel.from_pretrained(
        f"projects/retri/{config_path}.yaml",
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
        print(
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


def embed_pose(pose, model_name="default"):
    model_info = get_model(model_name)
    model = model_info["model"]

    caps, cmasks = preprocess_text("", model_name)
    poses = pose if isinstance(pose, list) else [pose]
    embeddings = []

    pose_frames_l = []
    for p in poses:
        pose_frames = preprocess_pose(p)
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
        buffer = f.read()
        return Pose.read(buffer, start_time=start_time_ms, end_time=end_time_ms)


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
        "--query-pose-dir",
        default="/opt/home/cleong/projects/semantic-sign-language-search/setup_signCLIP/fairseq/examples/MMPT/ASL_CITIZEN_GOD",
        type=Path,
        help="Path to the folder of query poses",
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
        default=None,
        help="Steps for slicing the pose",
    )

    parser.add_argument(
        "--models",
        type=str,
        default="default",
        # choices=[str(key) for key, _ in model_configs],
        help="Model to use (comma-separated)",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Model to use",
    )

    args = parser.parse_args()

    pose_path = args.pose_path

    print(args.models)

    if not pose_path.is_file():
        print(f"Error: File {pose_path} does not exist.")
        return
    # Define sliding window parameters
    window_size_ms = args.window_size_ms

    if window_size_ms is None:
        window_sizes = list(range(100, 1100, 100))
    else:
        window_sizes = [window_size_ms]

    pose_queries = list(args.query_pose_dir.glob("*.pose"))[:5]
    query_poses = []
    query_pose_embeddings = []
    for query_pose_path in pose_queries:
        print(f"Query: {query_pose_path}")
        query_pose = load_pose(query_pose_path)
        query_poses.append(query_pose)
        query_pose_embeddings.append(embed_pose(query_pose))

    models_to_use = args.models.split(",")

    print(models_to_use)

    # all_query_scores = defaultdict(list)

    for model_to_use in models_to_use:
        print(f"Loading model {model_to_use}")
        run_dir = args.results_dir / model_to_use / pose_path.stem
        plots_dir = run_dir / "search_plots"

        for window_size_ms in window_sizes:
            start_ms = args.start_time_ms or 0

            if args.end_time_ms is None:
                full_pose = load_pose(pose_path)  # DO NOT pass start/end
                duration_ms = int(
                    full_pose.body.fps
                    * 1000
                    * full_pose.body.data.shape[0]
                    / full_pose.body.fps
                )

                end_ms = duration_ms
            else:
                end_ms = args.end_time_ms

            plot_dir = plots_dir / "ASL_CITIZEN"
            plot_dir.mkdir(exist_ok=True, parents=True)
            out = (
                plot_dir
                / f"{args.query_pose_dir.name}_scores_{start_ms}_to_{end_ms}_step{window_size_ms}.png"
            )

            print(
                f"Sliding from {start_ms}ms to {end_ms}ms with step={window_size_ms}ms"
            )

            all_query_scores = defaultdict(list)

            for query_pose_path, query_embedding in tqdm(zip(
                pose_queries, query_pose_embeddings
            ), desc="Iterating over query poses"):
                print(f"Scoring with query {query_pose_path.name}")
                query_scores = []

                for window_start in tqdm(
                    range(start_ms, end_ms, window_size_ms),
                    desc=f"{query_pose_path.name} sliding...",
                ):
                    window_end = min(window_start + window_size_ms, end_ms)

                    try:
                        pose_window = load_pose(
                            args.pose_path,
                            start_time_ms=window_start,
                            end_time_ms=window_end,
                        )
                    except Exception as e:
                        print(
                            f"Failed to load window {window_start}-{window_end}ms: {e}"
                        )
                        continue

                    pose_window_embedding = embed_pose(pose_window, model_name=model_to_use)
                    score = np.matmul(pose_window_embedding, query_embedding.T).squeeze()
                    query_scores.append((window_start, window_end, score))

                all_query_scores[query_pose_path.name] = query_scores

            # --- Plotting ---
            plt.figure(figsize=(10, 5))

            for query_name, query_scores in all_query_scores.items():
                midpoints = [(start + end) // 2 for start, end, _ in query_scores]
                scores = [score for _, _, score in query_scores]
                plt.plot(midpoints, scores, marker="o", linestyle="-", label=query_name)

            plt.title(f"Scores over Sliding Windows (step={window_size_ms}ms)")
            plt.xlabel("Window Midpoint Time (ms)")
            plt.ylabel("Score")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out)
            print(out.resolve())


if __name__ == "__main__":
    main()
# python demo_sign_pose_inputs.py --window_size_ms 1000 --pose_path "/opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/samples/CBT-001-ase-2-Intro _ God Creates the World.pose" --models "asl_finetune_checkpoint_best" --start_time_ms 0 --end_time_ms 60000 --query-pose-dir ASL_CITIZEN_GOD
# python demo_sign_pose_inputs.py --window_size_ms 1000 --pose_path "/opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/samples/CBT-001-ase-2-Intro _ God Creates the World.pose" --models "asl_finetune_checkpoint_best" --start_time_ms 0 --end_time_ms 60000 --query-pose-dir ASL_CITIZEN_HEAVEN