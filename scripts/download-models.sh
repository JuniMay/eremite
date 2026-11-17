#!/bin/sh
# Downloads DeepSeek V4.1 Flash's files for eremite from Hugging Face: its
# checkpoint, and the routed experts of GGUF quantizations. The last shard of
# each GGUF set holds the Engram tables, which eremite reads from the
# checkpoint, so the script leaves that shard out. It installs Hugging Face's
# download command, hf, in a Python environment of its own, and deletes that
# environment at the end. Running the script again goes on where an
# interrupted download stopped.
#
# usage: scripts/download-models.sh [--dry-run] FOLDER [QUANT...]
#   FOLDER/deepseek-ai/DeepSeek-V4.1-Flash-1017 gets the checkpoint (510 GB).
#   FOLDER/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/QUANT gets each QUANT's
#   experts: UD-IQ2_XXS (165 GB, the default) or UD-IQ1_M (140 GB).
#   --dry-run lists the files that the download would fetch, and fetches none.
#
# HF_TOKEN gives a Hugging Face token to repositories that need one. It needs
# Python 3.10 or later. PYTHON names the Python to use (default: python3).
set -eu

# The huggingface_hub version whose hf command this script was tested with.
HUB_VERSION=2.4.0
CHECKPOINT=deepseek-ai/DeepSeek-V4.1-Flash-1017
EXPERTS=unsloth/DeepSeek-V4.1-Flash-1017-GGUF

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
