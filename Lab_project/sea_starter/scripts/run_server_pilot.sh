#!/usr/bin/env bash
# Run from sea_starter after overlaying the full-file ZIP.
# Each stage is explicit; test evaluation is never run during training.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STAGE="${1:-help}"
VLP_CFG="${VLP_CFG:-configs/server_pilot_vlp.yaml}"
BOUNDARY_CFG="${BOUNDARY_CFG:-configs/server_pilot_boundary.yaml}"
TRANSLATION_CFG="${TRANSLATION_CFG:-configs/server_pilot_translation.yaml}"
PREP_CFG="${PREP_CFG:-configs/server_pilot_preprocess.yaml}"
VLP_CKPT="${VLP_CKPT:-outputs/video_text/server_pilot/best_video_text.pt}"
BOUNDARY_CKPT="${BOUNDARY_CKPT:-outputs/sentence_boundaries_server_pilot/best_sentence_boundary.pt}"
TRANSLATION_CKPT="${TRANSLATION_CKPT:-outputs/translation/server_pilot_baseline/best_translation.pt}"
SPLIT_DIR="${SPLIT_DIR:-data/server_splits}"
mkdir -p outputs/reports/server_pilot

need_file() { [[ -f "$1" ]] || { echo "ERROR: required file missing: $1" >&2; exit 2; }; }
require_splits() {
  need_file "$SPLIT_DIR/train_ids.txt"
  need_file "$SPLIT_DIR/val_ids.txt"
  need_file "$SPLIT_DIR/test_ids.txt"
}

case "$STAGE" in
  split)
    ARGS=(--videos-dir "${VIDEOS_DIR:-data/videos}"
          --subtitles-dir "${SUBTITLES_DIR:-data/subtitles_raw}"
          --output-dir "$SPLIT_DIR" --seed "${SEED:-42}")
    if [[ -n "${METADATA:-}" ]]; then
      ARGS+=(--metadata "$METADATA" --group-by "${GROUP_BY:-recording}")
      [[ "${STRICT_SIGNER:-0}" == 1 ]] && ARGS+=(--strict-signer)
    fi
    [[ "${REQUIRE_CACHED:-0}" == 1 ]] && ARGS+=(--visual-cache-dir "${VISUAL_CACHE_DIR:-outputs/features}")
    [[ "${OVERWRITE_SPLITS:-0}" == 1 ]] && ARGS+=(--overwrite)
    python scripts/prepare_server_splits.py "${ARGS[@]}"
    ;;
  preflight)
    require_splits
    python -m persian_sea.cli.preflight \
      --config "$PREP_CFG" --video-ids "$SPLIT_DIR/all_ids.txt" \
      --report outputs/reports/server_pilot/preflight.json
    ;;
  preprocess)
    require_splits
    python -m persian_sea.cli.preprocess \
      --config "$PREP_CFG" --video-ids "$SPLIT_DIR/all_ids.txt" \
      --steps pose segment visual text
    ;;
  audit-caches)
    require_splits
    python scripts/audit_server_caches.py --config "$PREP_CFG" --split-dir "$SPLIT_DIR" \
      --output outputs/reports/server_pilot/cache_audit.json
    ;;
  vlp)
    require_splits
    python scripts/audit_server_caches.py --config "$PREP_CFG" --split-dir "$SPLIT_DIR" \
      --output outputs/reports/server_pilot/cache_audit.json
    persian-sea-train-video-text --config "$VLP_CFG"
    ;;
  boundary)
    require_splits
    need_file "$VLP_CKPT"
    python scripts/train_sentence_boundaries.py --config "$BOUNDARY_CFG"
    ;;
  translation-check)
    require_splits
    need_file "$VLP_CKPT"
    python -m persian_sea.translation.cli check --config "$TRANSLATION_CFG"
    ;;
  translation)
    require_splits
    need_file "$VLP_CKPT"
    python -m persian_sea.translation.cli check --config "$TRANSLATION_CFG"
    python -m persian_sea.translation.cli train --config "$TRANSLATION_CFG"
    ;;
  val)
    require_splits
    need_file "$TRANSLATION_CKPT"
    python -m persian_sea.translation.cli eval --config "$TRANSLATION_CFG" \
      --checkpoint "$TRANSLATION_CKPT" --split val \
      --output outputs/reports/server_pilot/translation_val.json
    ;;
  test)
    echo "WARNING: Run TEST only once with the chosen frozen checkpoint/config." >&2
    require_splits
    need_file "$TRANSLATION_CKPT"
    python -m persian_sea.translation.cli eval --config "$TRANSLATION_CFG" \
      --checkpoint "$TRANSLATION_CKPT" --split test \
      --output outputs/reports/server_pilot/translation_test_cue_guided.json
    ;;
  continuous)
    [[ -n "${VIDEO_ID:-}" ]] || { echo "Set VIDEO_ID=<heldout_video_stem>" >&2; exit 2; }
    need_file "$VLP_CKPT"
    need_file "$BOUNDARY_CKPT"
    need_file "$TRANSLATION_CKPT"
    python scripts/infer_continuous.py --translation-config "$TRANSLATION_CFG" \
      --translation-checkpoint "$TRANSLATION_CKPT" \
      --boundary-checkpoint "$BOUNDARY_CKPT" --video-id "$VIDEO_ID" \
      --output "outputs/reports/server_pilot/${VIDEO_ID}_continuous.json"
    ;;
  signshift-sea)
    [[ -n "${VIDEO_ID:-}" ]] || { echo "Set VIDEO_ID=<video_stem>" >&2; exit 2; }
    need_file "$BOUNDARY_CKPT"
    mkdir -p outputs/reports/server_pilot
    printf '%s\n' "$VIDEO_ID" > outputs/reports/server_pilot/one_video_id.txt
    python scripts/align_signshift.py \
      --config "$PREP_CFG" --checkpoint "$BOUNDARY_CKPT" \
      --video-ids outputs/reports/server_pilot/one_video_id.txt \
      --output-dir outputs/aligned_subtitles/server_signshift_pilot \
      --boundary-weight "${BOUNDARY_WEIGHT:-1.0}"
    ;;
  help|*)
    cat <<'TXT'
Usage: bash scripts/run_server_pilot.sh <stage>
Stages:
  split             Prepare held-out grouped splits (metadata optional)
  preflight         Audit raw inputs
  preprocess        Extract Pose, E4s, RGB/Visual and BGE text caches
  audit-caches      Check shape/timing/provenance of all visual/text caches
  vlp               VLP warm-up (train only; validate on val)
  boundary          Train SignShift-inspired sentence boundary head
  translation-check Validate files, VLP, splits and review constraints
  translation       Train mT5 translation pilot
  val               Cue-guided validation (not subtitle-free)
  test              Explicit held-out cue-guided test; only after model selection
  continuous        Subtitle-free inference (requires VIDEO_ID)
  signshift-sea     Generate SignShift-assisted SEA proposals (requires VIDEO_ID)

Set METADATA=data/video_metadata.csv and GROUP_BY=signer to enforce identity groups.
Set OVERWRITE_SPLITS=1 only intentionally; changing the split invalidates prior results.
TXT
    ;;
esac
