import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Export top matched original filenames to a text file."
    )
    parser.add_argument(
        "--json",
        default="outputs/result.json",
        help="Path to the matching result JSON file.",
    )
    parser.add_argument(
        "--out", default="outputs/out.txt", help="Path to the output text file."
    )
    parser.add_argument(
        "--full-path",
        action="store_true",
        help="Output absolute paths instead of just filenames.",
    )
    args = parser.parse_args()

    json_path = Path(args.json)
    out_path = Path(args.out)

    if not json_path.exists():
        print(f"[Error] File not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    extracted_files = []

    for selected_path, matches in data.items():
        if not matches:
            print(f"[WARN] No matches found for: {selected_path}")
            continue

        top1_original_path = matches[0]["original_path"]
        p = Path(top1_original_path)
        if args.full_path:
            extracted_files.append(str(p))
        else:
            extracted_files.append(p.name)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for fname in extracted_files:
            f.write(fname + "\n")

    print(
        f"Successfully exported {len(extracted_files)} files to: {out_path.resolve()}"
    )

    # Generate a string query for photo catalog software
    no_ext_names = [Path(f).stem for f in extracted_files]
    print("\n" + "=" * 80)
    print(
        "【Copy & Paste into Lightroom / Capture One Search Bar】 (Space-separated, no extensions):"
    )
    print(" ".join(no_ext_names))
    print("=" * 80)


if __name__ == "__main__":
    main()
