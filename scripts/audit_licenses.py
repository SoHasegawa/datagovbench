"""Audit per-dataset licenses across a DataGovBench dataset directory.

Walks every metadata.json under DATASET_DIR, classifies the `license` field
into one of {ok, attribution, sharealike, review, problem}, and either prints
a human-readable report or emits content suitable for LICENSES_THIRD_PARTY.md.

Examples
--------
  python scripts/audit_licenses.py /data/shasegawa/opendatabench
  python scripts/audit_licenses.py /data/shasegawa/opendatabench --strict
  python scripts/audit_licenses.py /data/shasegawa/opendatabench --markdown > LICENSES_THIRD_PARTY.md
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


LICENSE_CATEGORIES = {
    # Public domain — no attribution required
    "us-pd": "ok",
    "Creative Commons CCZero": "ok",
    "Other (Public Domain)": "ok",
    "Open Data Commons Public Domain Dedication and License (PDDL)": "ok",

    # Permissive with attribution
    "Creative Commons Attribution": "attribution",
    "Creative Commons Attribution 4.0": "attribution",
    "Creative Commons Attribution 4.0 International": "attribution",
    "Creative Commons 4.0 Attribution (CC-BY) licence – Quebec": "attribution",
    "Open Government Licence - Canada": "attribution",
    "Open Government Licence - Alberta": "attribution",
    "Open Government Licence – Nova Scotia": "attribution",
    "UK Open Government Licence (OGL)": "attribution",
    "Open Government Licence – Ontario": "attribution",
    "Open Government Licence - British Columbia": "attribution",
    "Open Government License - City of Surrey": "attribution",
    "open-government-licence-toronto": "attribution",
    "Open Data Commons Attribution License": "attribution",
    "ocd-by": "attribution",

    # ShareAlike — redistribution OK but propagates the SA constraint
    "Creative Commons Attribution Share-Alike": "sharealike",

    # NonCommercial — accepted under the benchmark's non-commercial policy,
    # but the whole benchmark inherits the NC constraint.
    "CC-BY-NC-SA-4.0": "noncommercial",

    # Problematic — unknown terms
    "License not specified": "problem",

    # Opaque source flag, needs per-dataset inspection
    "other-license-specified": "review",
}


def classify(license_value):
    if license_value is None or license_value == "" or license_value == "<MISSING>":
        return "problem"
    if license_value in LICENSE_CATEGORIES:
        return LICENSE_CATEGORIES[license_value]
    # Heuristics for license strings we haven't seen literally
    if "NC" in license_value:
        return "noncommercial"
    if "Share-Alike" in license_value or "ShareAlike" in license_value or "-SA" in license_value:
        return "sharealike"
    if license_value.startswith("CalHHS Terms of Use"):
        # Custom terms that permit redistribution with attribution; documented inline
        return "attribution"
    return "review"


def scan(dataset_dir):
    root = Path(dataset_dir)
    by_license = defaultdict(list)
    for p in sorted(root.glob("**/metadata.json")):
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            print(f"WARN: could not parse {p}: {e}", file=sys.stderr)
            continue
        license_value = data.get("license") or "<MISSING>"
        rel = p.relative_to(root).parent
        by_license[license_value].append({
            "path": str(rel),
            "publisher": data.get("publisher", ""),
            "title": data.get("dataset_title", ""),
            "landingPage": data.get("landingPage") or "",
        })
    return dict(by_license)


def print_report(by_license):
    total = sum(len(v) for v in by_license.values())
    counts = defaultdict(int)
    for lic, items in by_license.items():
        counts[classify(lic)] += len(items)

    print(f"Total datasets: {total}")
    print()
    print("By category:")
    for cat in ("ok", "attribution", "sharealike", "noncommercial", "review", "problem"):
        print(f"  {cat:13s} {counts[cat]:4d}")
    print()
    print("By license string:")
    for lic, items in sorted(by_license.items(), key=lambda kv: -len(kv[1])):
        print(f"  [{classify(lic):11s}] {len(items):4d}  {lic}")


def format_markdown(by_license):
    out = []
    out.append("# Third-Party Dataset Licenses")
    out.append("")
    out.append("This benchmark aggregates publicly available datasets from open-data portals.")
    out.append("Each dataset retains its original license; this file enumerates them.")
    out.append("The benchmark code and packaging are licensed separately (see LICENSE).")
    out.append("")

    nc_count = sum(len(v) for k, v in by_license.items() if classify(k) == "noncommercial")
    if nc_count:
        out.append("> **Important — NonCommercial.** Because this benchmark includes one or more")
        out.append("> datasets under a NonCommercial license (e.g. CC-BY-NC-SA-4.0), the benchmark")
        out.append("> as a whole may only be used for non-commercial purposes. Commercial use of")
        out.append("> the benchmark — including evaluation of commercial products — is prohibited")
        out.append("> unless those datasets are removed from your copy.")
        out.append("")

    out.append("---")
    out.append("")
    for lic, items in sorted(by_license.items(), key=lambda kv: -len(kv[1])):
        category = classify(lic)
        out.append(f"## {lic}")
        out.append("")
        if category == "noncommercial":
            out.append("> **NonCommercial + ShareAlike.** Use of this dataset for commercial purposes")
            out.append("> is prohibited. Redistributions must be released under a compatible")
            out.append("> NonCommercial-ShareAlike license.")
            out.append("")
        if category == "sharealike":
            out.append("> **ShareAlike.** Redistributions of this dataset (modified or not) must be")
            out.append("> released under a compatible ShareAlike license.")
            out.append("")
        if category == "problem":
            out.append("> **Non-standard license.** See the source URL on each item for the full terms.")
            out.append("")
        if lic.startswith("CalHHS Terms of Use"):
            out.append("> **Revocable license.** The CalHHS Terms of Use grant a *revocable*")
            out.append("> non-exclusive license. Attribution to the specific CalHHS department,")
            out.append("> citation to the source webpage, and date of publication are required.")
            out.append("")
        out.append(f"({len(items)} dataset{'s' if len(items) != 1 else ''})")
        out.append("")
        for item in items:
            bullet = f"- **{item['title']}** — {item['publisher']}"
            if item["landingPage"]:
                bullet += f" — [source]({item['landingPage']})"
            out.append(bullet)
            out.append(f"  `{item['path']}`")
        out.append("")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dataset_dir", help="root directory containing per-dataset metadata.json files")
    parser.add_argument("--markdown", action="store_true",
                        help="emit LICENSES_THIRD_PARTY.md content on stdout")
    parser.add_argument("--strict", action="store_true",
                        help="exit nonzero if any 'problem' license is present")
    args = parser.parse_args()

    root = Path(args.dataset_dir)
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 2

    by_license = scan(root)
    if not by_license:
        print(f"ERROR: no metadata.json files found under {root}", file=sys.stderr)
        return 2

    if args.markdown:
        print(format_markdown(by_license))
    else:
        print_report(by_license)

    if args.strict:
        problems = sum(len(v) for k, v in by_license.items() if classify(k) == "problem")
        if problems:
            print(f"\nFAIL: {problems} dataset(s) carry a problem license.", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
