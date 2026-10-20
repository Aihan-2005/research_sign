
import argparse
import json
from persian_sea.data.manifest import read_manifest, audit_manifest, write_id_files


def main():
    ap = argparse.ArgumentParser(description="Validate manually assigned train/val/test splits")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--strict-signer", action="store_true")
    ap.add_argument("--write", action="store_true", help="Only write IDs when explicitly requested")
    ap.add_argument("--overwrite", action="store_true")
    opts = ap.parse_args()
    rec = read_manifest(opts.manifest)
    report = audit_manifest(rec, opts.strict_signer)
    if opts.write:
        report["written"] = write_id_files(rec, opts.out_dir, overwrite=opts.overwrite)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
