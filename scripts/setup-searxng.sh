#!/bin/sh
# Sets up eremite's own SearXNG, the search engine behind Codex's web tool:
# SearXNG's code at a fixed commit, a Python environment with its packages and
# granian, and eremite's settings for it. The server starts this SearXNG when a
# search comes, and stops it when searches stop. Running the script again
# brings the folder to the same commit and keeps its settings.
#
# usage: scripts/setup-searxng.sh [FOLDER]
#   FOLDER is where SearXNG goes, by default ~/.local/share/eremite/searxng:
#   the server's default searxng_dir. For another FOLDER, set the server's
#   searxng_dir to it.
#
# It needs git and Python 3.10 or later. PYTHON names the Python to use
# (default: python3). It downloads about 100 MB.
set -eu

# The SearXNG commit that eremite's web tool was tested with.
COMMIT=ac11e14c5edf995d4d2f066bfb885a614f52a978
REPOSITORY=https://github.com/searxng/searxng.git

dir=${1:-$HOME/.local/share/eremite/searxng}
python=${PYTHON:-python3}

fail() {
  echo "setup-searxng: $*" >&2
  exit 1
}

command -v git > /dev/null 2>&1 ||
  fail "git is missing: install Apple's command line tools (xcode-select --install)"
command -v "$python" > /dev/null 2>&1 ||
  fail "$python is missing: install Python 3.10 or later, or set PYTHON to one"
"$python" -c 'import sys; sys.exit(sys.version_info < (3, 10))' ||
  fail "$python is older than 3.10: install a newer Python, or set PYTHON to one"

echo "SearXNG $COMMIT in $dir"
mkdir -p "$dir"

# SearXNG's code at the commit, without its history.
if [ ! -d "$dir/src/.git" ]; then
  git init -q "$dir/src"
  git -C "$dir/src" remote add origin "$REPOSITORY"
fi
git -C "$dir/src" fetch -q --depth 1 origin "$COMMIT"
git -C "$dir/src" checkout -q --detach "$COMMIT"

# A Python environment with SearXNG's packages, granian (the server that runs
# it), and SearXNG itself, so that granian can import it.
[ -x "$dir/venv/bin/python" ] || "$python" -m venv "$dir/venv"
"$dir/venv/bin/python" -m pip install -q --upgrade pip setuptools wheel
"$dir/venv/bin/python" -m pip install -q \
  -r "$dir/src/requirements.txt" -r "$dir/src/requirements-server.txt"
"$dir/venv/bin/python" -m pip install -q --no-build-isolation --no-deps \
  -e "$dir/src"

# eremite's settings, written once with a secret key of their own. The server
# runs SearXNG on a Unix socket that only this user may use.
if [ ! -f "$dir/settings.yml" ]; then
  key=$("$dir/venv/bin/python" -c 'import secrets; print(secrets.token_hex(32))')
  umask 077
  cat > "$dir/settings.yml" << EOF
# eremite's SearXNG, for Codex's web tool. These override SearXNG's defaults.
use_default_settings: true
general:
  instance_name: "eremite"
search:
  formats: [html, json]
server:
  secret_key: "$key"
  limiter: false
  public_instance: false
  image_proxy: false
engines:
  # DuckDuckGo answers automated queries with a CAPTCHA.
  - name: duckduckgo
    disabled: true
EOF
fi

echo "SearXNG is set up in $dir"
