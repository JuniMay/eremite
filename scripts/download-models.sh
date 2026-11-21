#!/bin/sh
# Downloads the models that eremite serves from Hugging Face: DeepSeek V4.1
# Flash's checkpoint, the routed experts of GGUF quantizations, Thinking
# Machines Lab's Inkling-2-Mini checkpoint, LiquidAI's d1-2B decision model,
# and MiniMax-H3's image and video metadata, tokenizers and Comfy-Org quantized
# checkpoints. The last shard of each GGUF set holds the Engram tables, which
# eremite reads from the checkpoint, so the script leaves that shard out. It
# installs Hugging Face's download command, hf, in a Python environment of its
# own, and deletes that environment at the end. Running the script again goes
# on where an interrupted download stopped.
#
# usage: scripts/download-models.sh [--dry-run] FOLDER [QUANT...]
#   FOLDER/deepseek-ai/DeepSeek-V4.1-Flash-1017 gets the checkpoint (510 GB).
#   FOLDER/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/QUANT gets each QUANT's
#   experts: UD-IQ2_XXS (165 GB, the default) or UD-IQ1_M (140 GB).
#   FOLDER/thinkingmachines/Inkling-2-Mini gets Inkling-2-Mini (20.2 GB,
#   including its MTP draft head).
#   FOLDER/LiquidAI/d1-2B gets d1-2B (3.9 GB).
#   FOLDER/MiniMaxAI/MiniMax-H3-Image gets MiniMax-H3-Image's metadata and
#   tokenizer (23 MB).
#   FOLDER/MiniMaxAI/MiniMax-H3 gets MiniMax-H3 Video's metadata and processor
#   tokenizer (23 MB).
#   FOLDER/Comfy-Org/MiniMax-H3 gets its Qwen3-VL-32B NVFP4 text and vision
#   encoder (15.7 GB), FL2VA and REF2VA pruned INT8-ConvRot DiTs (21.0 GB
#   each), INT8-ConvRot Video VAE (2.8 GB), FP32 Audio VAE (0.6 GB), and BF16
#   Turbo LoRAs (2.0 GB each).
#   FOLDER/Comfy-Org/MiniMax-H3-Image gets its pruned INT8-ConvRot DiT
#   (21.0 GB) and FP16 image VAE (1.2 GB).
#   --dry-run lists the files that the download would fetch, and fetches none.
#
# HF_TOKEN gives a Hugging Face token to repositories that need one. It needs
# Python 3.10 or later. PYTHON names the Python to use (default: python3).
set -eu

# The huggingface_hub version whose hf command this script was tested with.
HUB_VERSION=2.4.0
CHECKPOINT=deepseek-ai/DeepSeek-V4.1-Flash-1017
EXPERTS=unsloth/DeepSeek-V4.1-Flash-1017-GGUF
INKLING2=thinkingmachines/Inkling-2-Mini
DECISIONS=LiquidAI/d1-2B
MINIMAX_H3_IMAGE=MiniMaxAI/MiniMax-H3-Image
MINIMAX_H3_VIDEO=MiniMaxAI/MiniMax-H3
MINIMAX_H3_COMFY=Comfy-Org/MiniMax-H3
MINIMAX_H3_IMAGE_COMFY=Comfy-Org/MiniMax-H3-Image
MINIMAX_H3_TEXT_ENCODER=text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
MINIMAX_H3_IMAGE_DIT=diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors
MINIMAX_H3_IMAGE_VAE=vae/minimax_h3_image_vae_fp16.safetensors
MINIMAX_H3_VIDEO_FL2VA_DIT=diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
MINIMAX_H3_VIDEO_REF2VA_DIT=diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors
MINIMAX_H3_VIDEO_VAE=vae/minimax_h3_video_vae_int8_convrot.safetensors
MINIMAX_H3_AUDIO_VAE=vae/minimax_h3_audio_vae_fp32.safetensors
MINIMAX_H3_FL2V_LORA_4=loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors
MINIMAX_H3_FL2V_LORA_8=loras/minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors
MINIMAX_H3_REF2V_LORA_4=loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors

fail() {
  echo "download-models: $*" >&2
  exit 1
}

dry_run=
if [ "${1:-}" = --dry-run ]; then
  dry_run=--dry-run
  shift
fi
[ $# -ge 1 ] || fail "usage: scripts/download-models.sh [--dry-run] FOLDER [QUANT...]"
folder=$1
shift
[ $# -ge 1 ] || set -- UD-IQ2_XXS
python=${PYTHON:-python3}

command -v "$python" > /dev/null 2>&1 ||
  fail "$python is missing: install Python 3.10 or later, or set PYTHON to one"
"$python" -c 'import sys; sys.exit(sys.version_info < (3, 10))' ||
  fail "$python is older than 3.10: install a newer Python, or set PYTHON to one"

# hf in a Python environment of its own, deleted at the end.
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
"$python" -m venv "$work/venv"
"$work/venv/bin/python" -m pip install -q "huggingface_hub==$HUB_VERSION"
hf=$work/venv/bin/hf

"$hf" download "$CHECKPOINT" --local-dir "$folder/$CHECKPOINT" $dry_run
for quant in "$@"; do
  # A set's last shard, whose number is its count of shards, for 2 to 9 shards.
  last_shards=
  for count in 2 3 4 5 6 7 8 9; do
    last_shards="$last_shards --exclude $quant/*-0000$count-of-0000$count.gguf"
  done
  # The patterns stay unexpanded: a GGUF shard's name has no space.
  set -f
  "$hf" download "$EXPERTS" --include "$quant/*" $last_shards \
    --local-dir "$folder/$EXPERTS" $dry_run
  set +f
done
"$hf" download "$INKLING2" --local-dir "$folder/$INKLING2" $dry_run
"$hf" download "$DECISIONS" --local-dir "$folder/$DECISIONS" $dry_run
# MiniMax-H3 image and video metadata, tokenizers, and Comfy-Org weights.
set -f
"$hf" download "$MINIMAX_H3_IMAGE" \
  --include model_index.json \
  --include pdd_config.json \
  --include "tokenizer/*" \
  --include "processor/*" \
  --include transformer/config.json \
  --include vae/config.json \
  --local-dir "$folder/$MINIMAX_H3_IMAGE" $dry_run
"$hf" download "$MINIMAX_H3_VIDEO" \
  --include model_index.json \
  --include "FL2VA/model_index.json" \
  --include "Ref2VA/model_index.json" \
  --include "processor/*" \
  --local-dir "$folder/$MINIMAX_H3_VIDEO" $dry_run
"$hf" download "$MINIMAX_H3_COMFY" \
  --include "$MINIMAX_H3_TEXT_ENCODER" \
  --include "$MINIMAX_H3_VIDEO_FL2VA_DIT" \
  --include "$MINIMAX_H3_VIDEO_REF2VA_DIT" \
  --include "$MINIMAX_H3_VIDEO_VAE" \
  --include "$MINIMAX_H3_AUDIO_VAE" \
  --include "$MINIMAX_H3_FL2V_LORA_4" \
  --include "$MINIMAX_H3_FL2V_LORA_8" \
  --include "$MINIMAX_H3_REF2V_LORA_4" \
  --local-dir "$folder/$MINIMAX_H3_COMFY" $dry_run
"$hf" download "$MINIMAX_H3_IMAGE_COMFY" \
  --include "$MINIMAX_H3_IMAGE_DIT" \
  --include "$MINIMAX_H3_IMAGE_VAE" \
  --local-dir "$folder/$MINIMAX_H3_IMAGE_COMFY" $dry_run
set +f
