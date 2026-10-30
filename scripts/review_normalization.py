"""Open private normalization worksheets in a browser at http://127.0.0.1:8001."""
import argparse
from pathlib import Path

import uvicorn

from symptom_rec.kb.loader import load_kb
from symptom_rec.normalize.review_web import create_review_app
from symptom_rec.paths import BUILD_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=BUILD_DIR / "normalization_review")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to(BUILD_DIR.resolve()):
        raise SystemExit("Private review sheets must stay under ignored build/")
    if not (directory / "manifest.json").exists():
        raise SystemExit("Run make normalization-review first")
    app = create_review_app(directory, load_kb())
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
