#!/usr/bin/env bash
# Download and unpack RAVDESS Emotional Song Audio from Kaggle.
#
#   bash scripts/download_dataset.sh [target_dir]
#
# Requires a Kaggle API token at ~/.kaggle/kaggle.json (chmod 600).
# Get one from https://www.kaggle.com/settings  ->  "Create New Token".
set -euo pipefail

TARGET="${1:-data/ravdess}"
SLUG="uwrfkaggler/ravdess-emotional-song-audio"

command -v kaggle >/dev/null 2>&1 || { echo "kaggle CLI not found. pip install kaggle"; exit 1; }
[ -f "${HOME}/.kaggle/kaggle.json" ] || { echo "missing ${HOME}/.kaggle/kaggle.json"; exit 1; }
chmod 600 "${HOME}/.kaggle/kaggle.json"

mkdir -p "${TARGET}"
echo ">> downloading ${SLUG} (~456 MB) into ${TARGET}"
kaggle datasets download "${SLUG}" -p "${TARGET}" --unzip

RAW=$(find "${TARGET}" -name '*.wav' | wc -l)
UNIQUE=$(find "${TARGET}" -name '*.wav' -exec basename {} \; | sort -u | wc -l)

cat <<EOF

   archive entries : ${RAW}
unique recordings : ${UNIQUE}

EOF

if [ "${RAW}" -ne "${UNIQUE}" ]; then
  cat <<'EOF'
NOTE: this mirror ships every recording twice (once under Actor_XX/ and
once under audio_song_actors_01-24/Actor_XX/). Training on the raw file
list leaks duplicates across the train/test split.

anxiety_speech.data.build_index() de-duplicates by default — nothing more
to do. See docs/DATASET.md.
EOF
fi

echo ">> dataset ready at ${TARGET}"
