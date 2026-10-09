"""Download the pinned public dataset. Raw files stay in Git-ignored data/."""

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw"
MAX_ARCHIVE_BYTES = 5_000_000


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    manifest = json.loads((PROJECT / "dataset.json").read_text())
    expected_files = manifest["files"]

    if all(
        (RAW / name).is_file() and sha256((RAW / name).read_bytes()) == expected
        for name, expected in expected_files.items()
    ):
        print("Dataset already downloaded; checksums match.")
        return

    with urllib.request.urlopen(manifest["archive_url"], timeout=30) as response:
        archive = response.read(MAX_ARCHIVE_BYTES + 1)
    if len(archive) > MAX_ARCHIVE_BYTES:
        raise SystemExit("Archive exceeds the expected size limit.")
    if sha256(archive) != manifest["archive_sha256"]:
        raise SystemExit("Archive checksum changed. Inspect the source before updating the manifest.")

    # Read only the named files; verify everything before writing raw data.
    contents = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        for name, expected in expected_files.items():
            data = bundle.read(name)
            if sha256(data) != expected:
                raise SystemExit(f"Checksum mismatch: {name}")
            contents[name] = data

    RAW.mkdir(parents=True, exist_ok=True)
    for name, data in contents.items():
        (RAW / name).write_bytes(data)
        print(f"Saved data/raw/{name} ({len(data):,} bytes; checksum verified)")


if __name__ == "__main__":
    main()
