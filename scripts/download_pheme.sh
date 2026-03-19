#!/usr/bin/env bash

set -euo pipefail

RAW_ROOT="${RAW_ROOT:-data/raw/pheme}"
ARCHIVE_PATH="${ARCHIVE_PATH:-}"
PHEME_URL="${PHEME_URL:-}"

mkdir -p "$RAW_ROOT"

print_manual_instructions() {
  cat <<EOF
PHEME raw-data setup requires one of the following:

1. Automatic download:
   PHEME_URL="<dataset archive url>" bash scripts/download_pheme.sh

2. Manual archive placement:
   ARCHIVE_PATH="/path/to/pheme.zip" bash scripts/download_pheme.sh

3. Manual extracted dataset placement:
   Copy the extracted PHEME event folders into:
   $RAW_ROOT

Expected layout after setup:
  $RAW_ROOT/<event_name>/<thread_id>/source-tweets/*.json
  $RAW_ROOT/<event_name>/<thread_id>/reactions/*.json
  $RAW_ROOT/<event_name>/<thread_id>/structure.json

The script does not invent a download URL because PHEME distribution sources vary.
If you already placed the dataset manually, continue with:
  python scripts/verify_pheme_layout.py
  python scripts/prepare_pheme.py
EOF
}

has_existing_layout() {
  find "$RAW_ROOT" -mindepth 3 -maxdepth 3 -type d -name source-tweets | grep -q .
}

extract_archive() {
  local archive="$1"
  echo "Extracting archive into $RAW_ROOT"

  case "$archive" in
    *.zip)
      unzip -q -o "$archive" -d "$RAW_ROOT"
      ;;
    *.tar.gz|*.tgz)
      tar -xzf "$archive" -C "$RAW_ROOT"
      ;;
    *.tar)
      tar -xf "$archive" -C "$RAW_ROOT"
      ;;
    *)
      echo "Unsupported archive format: $archive" >&2
      exit 1
      ;;
  esac
}

download_archive() {
  local url="$1"
  local tmp_archive
  tmp_archive="$(mktemp /tmp/pheme_download.XXXXXX)"

  echo "Downloading PHEME archive from $url"
  if command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$tmp_archive"
  elif command -v wget >/dev/null 2>&1; then
    wget "$url" -O "$tmp_archive"
  else
    echo "Neither curl nor wget is available for automatic download." >&2
    rm -f "$tmp_archive"
    exit 1
  fi

  local detected_archive="$tmp_archive"
  case "$url" in
    *.zip)
      detected_archive="${tmp_archive}.zip"
      mv "$tmp_archive" "$detected_archive"
      ;;
    *.tar.gz|*.tgz)
      detected_archive="${tmp_archive}.tar.gz"
      mv "$tmp_archive" "$detected_archive"
      ;;
    *.tar)
      detected_archive="${tmp_archive}.tar"
      mv "$tmp_archive" "$detected_archive"
      ;;
    *)
      echo "Downloaded file extension is unknown; set ARCHIVE_PATH manually if extraction fails." >&2
      rm -f "$tmp_archive"
      exit 1
      ;;
  esac

  extract_archive "$detected_archive"
  rm -f "$detected_archive"
}

if has_existing_layout; then
  echo "Detected an existing PHEME-like layout under $RAW_ROOT"
  echo "Skipping download. Next step: python scripts/verify_pheme_layout.py"
  exit 0
fi

if [[ -n "$ARCHIVE_PATH" ]]; then
  if [[ ! -f "$ARCHIVE_PATH" ]]; then
    echo "Archive not found: $ARCHIVE_PATH" >&2
    exit 1
  fi
  extract_archive "$ARCHIVE_PATH"
elif [[ -n "$PHEME_URL" ]]; then
  download_archive "$PHEME_URL"
else
  print_manual_instructions
  exit 0
fi

echo "Download/setup step completed."
echo "Run the verifier next:"
echo "  python scripts/verify_pheme_layout.py"
