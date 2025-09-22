#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# POSE_PATHS=(
#     "/data/petabyte/cleong/data/DBL_Deaf_Bibles/webdataset_extracted/ase/chronological_bible_translation_in_american_sign_language_119_introductions_and_passages/ase_chronological_bible_translation_in_american_sign_language_119_introductions_and_passages_cbt-003-ase-3-passage _ the first man and woman disobey god.pose-mediapipe.pose"
# )


# POSE_PATHS=(
#     "/opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/cbt033_flipped/cbt_033_flipped.pose"
# )
mapfile -t POSE_PATHS < <(
    find "/data/petabyte/cleong/data/DBL_Deaf_Bibles/webdataset_extracted/ase/chronological_bible_translation_in_american_sign_language_119_introductions_and_passages/" -name "*-passage*.pose"
)
echo "$POSE_PATHS"

# POSE_PATHS=(
#     "/data/petabyte/cleong/data/DBL_Deaf_Bibles/webdataset_extracted/ase/chronological_bible_translation_in_american_sign_language_119_introductions_and_passages/ase_chronological_bible_translation_in_american_sign_language_119_introductions_and_passages_cbt-003-ase-3-passage _ the first man and woman disobey god.pose-mediapipe.pose"
# )


# POSE_PATHS=(
#     "/data/petabyte/cleong/data/DBL_Deaf_Bibles/webdataset_extracted/ase/chronological_bible_translation_in_american_sign_language_119_introductions_and_passages/ase_chronological_bible_translation_in_american_sign_language_119_introductions_and_passages_cbt-033-ase-2-passage _ israel fails to conquer ai.pose-mediapipe.pose"
# )

# POSE_PATHS=(
#     "/data/petabyte/cleong/data/DBL_Deaf_Bibles/webdataset_extracted/ase/chronological_bible_translation_in_american_sign_language_119_introductions_and_passages/ase_chronological_bible_translation_in_american_sign_language_119_introductions_and_passages_cbt-001-ase-3-passage _ god creates the world.pose-mediapipe.pose"
# )

# SAMPLES=(0 5 15 25)
SAMPLES=(0 25)
# SAMPLES=(0)

# STEP_WINDOW_PAIRS=(
#     "100:500"
#     "1500:3000"
#     "200:1000"
#     "1000:6000"
# )


STEP_WINDOW_PAIRS=(
    "100:500"
)

SCRIPT="/opt/home/cleong/projects/semantic-sign-language-search/setup_signCLIP/fairseq/examples/MMPT/search_with_text_and_poses.py"

#         --eng "SILVER ISRAEL MAN FAMILY SON COME PEOPLE SELECT TENT ENEMY HIDE BURN TROUBLE BAR BRING GOD" \
        # --eng "DAY EARTH GOD HEAVEN" \
        # --eng "GOD KNOW MAN TREE" \

# Generate all combinations safely (tab separated)
for pose in "${POSE_PATHS[@]}"; do
    for spg in "${SAMPLES[@]}"; do
        for pair in "${STEP_WINDOW_PAIRS[@]}"; do
            IFS=":" read -r step window <<< "$pair"
            printf "%s\t%s\t%s\t%s\n" "$pose" "$spg" "$step" "$window"
        done
    done
done | parallel --progress -j1 --colsep '\t' '
    echo "Running: pose_path={1}, samples-per-gloss={2}, step_size_ms={3}, window_size_ms={4}"
    python '"$SCRIPT"' \
        --pose_path {1} \
        --start_time_ms 0 \
        --step_size_ms {3} \
        --window_size_ms {4} \
        --eng "SILVER ISRAEL MAN FAMILY SON COME PEOPLE SELECT TENT ENEMY HIDE BURN TROUBLE BAR BRING GOD" \
        --model "asl_finetune_checkpoint_best" \
        --samples-per-gloss {2}
'
