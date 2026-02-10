#!/usr/bin/env python3
"""
Generate a doc-traceable coverage matrix for `research/privacy_new.txt`.

Deterministic + offline:
- No network
- Deterministic output independent of invocation paths
- Designed for CI gating: `--check` fails if regeneration would change outputs

Coverage goals:
1) atomize *every* item into stable requirement atoms:
   - headings (markdown, numeric, letter headings)
   - paragraph blocks (split by blank lines)
   - list items (best-effort)
   - table blocks (best-effort)
   - code-fence blocks
2) emit:
   - `privacy_new_atoms.json` (machine source of truth)
   - `privacy_new_coverage_matrix.md` (human report)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


RE_HEADING_MD = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$")
RE_HEADING_NUM_PAREN = re.compile(r"^\s*(?P<num>\d+(?:\.\d+)*)\)\s+(?P<title>.+?)\s*$")
RE_HEADING_NUM_DOT = re.compile(r"^\s*(?P<num>\d+(?:\.\d+)+)\s+(?P<title>.+?)\s*$")
RE_HEADING_LETTER_PAREN = re.compile(r"^\s*(?P<let>[A-Z])\)\s+(?P<title>.+?)\s*$")
RE_LIST_BULLET = re.compile(r"^\s*[-*+]\s+(?P<body>.+?)\s*$")
RE_LIST_NUM = re.compile(r"^\s*(?P<n>\d+)\.\s+(?P<body>.+?)\s*$")
RE_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")

DS_REQ = b"uvcc.reqatom.v1\0"
DS_GROUP = b"uvcc.reqgroup.v1\0"


@dataclass(frozen=True)
class Atom:
    idx: int
    req_id: str
    group_id: str
    kind: str  # heading|paragraph|list_item|table_block|code_block
    start_line: int
    end_line: int
    section_path: Tuple[str, ...]
    text: str


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_lines(path: Path) -> List[str]:
    # Preserve exact line content (minus trailing newline).
    return path.read_text(encoding="utf-8", errors="strict").splitlines()


def _relpath_if_possible(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _heading_match(line: str) -> Optional[Tuple[int, str]]:
    """
    Returns (level, title) if line looks like a heading, else None.

    - Markdown headings: #..###### => level = number of '#'
    - Numeric headings:
      - `1) Title` => level 1
      - `4.2 Title` => level 2
      - `4.2.1) Title` => level 3
    - Letter headings:
      - `A) Title` => level 1
    """
    m = RE_HEADING_MD.match(line)
    if m:
        lvl = min(6, len(m.group("hashes")))
        return (lvl, _collapse_ws(m.group("title")))
    m = RE_HEADING_NUM_PAREN.match(line)
    if m:
        dots = m.group("num").count(".")
        return (1 + dots, _collapse_ws(m.group("title")))
    m = RE_HEADING_NUM_DOT.match(line)
    if m:
        dots = m.group("num").count(".")
        return (1 + dots, _collapse_ws(m.group("title")))
    m = RE_HEADING_LETTER_PAREN.match(line)
    if m:
        return (1, _collapse_ws(f"{m.group('let')}) {m.group('title')}"))
    return None


def _stable_req_id(*, kind: str, section_path: Sequence[str], text: str) -> str:
    """
    Stable id derived from semantic content, not line numbers.
    """
    h = hashlib.sha256()
    h.update(DS_REQ)
    h.update(kind.encode("utf-8"))
    h.update(b"\n")
    for t in section_path:
        h.update(_collapse_ws(t).encode("utf-8"))
        h.update(b"\n")
    if kind == "code_block":
        h.update(text.encode("utf-8"))
    else:
        h.update(_collapse_ws(text).encode("utf-8"))
    return "uvcc_req_" + h.hexdigest()[:24]


def _stable_group_id(section_path: Sequence[str]) -> str:
    if not section_path:
        return "uvcc_group_root"
    h = hashlib.sha256()
    h.update(DS_GROUP)
    for t in section_path:
        h.update(_collapse_ws(t).encode("utf-8"))
        h.update(b"\n")
    return "uvcc_group_" + h.hexdigest()[:16]


def atomize_privacy_new(lines: List[str]) -> List[Atom]:
    in_code = False
    section_stack: List[Tuple[int, str]] = []  # (level, title)
    atoms: List[Atom] = []

    para_lines: List[str] = []
    para_start: Optional[int] = None

    code_lines: List[str] = []
    code_start: Optional[int] = None

    # Used to disambiguate identical ids from identical content.
    seen: Dict[str, int] = {}

    def cur_section_path() -> Tuple[str, ...]:
        return tuple(t for _, t in section_stack)

    def _emit(kind: str, start_line: int, end_line: int, text: str) -> None:
        rid0 = _stable_req_id(kind=kind, section_path=cur_section_path(), text=text)
        n = seen.get(rid0, 0) + 1
        seen[rid0] = n
        rid = rid0 if n == 1 else f"{rid0}-{n}"
        gid = _stable_group_id(cur_section_path())
        atoms.append(
            Atom(
                idx=len(atoms),
                req_id=rid,
                group_id=gid,
                kind=kind,
                start_line=int(start_line),
                end_line=int(end_line),
                section_path=cur_section_path(),
                text=text,
            )
        )

    def flush_para(end_line: int) -> None:
        nonlocal para_lines, para_start
        if para_start is None:
            return
        text = "\n".join(para_lines).rstrip()
        if text.strip():
            _emit("paragraph", int(para_start), int(end_line), text)
        para_lines = []
        para_start = None

    def flush_code(end_line: int) -> None:
        nonlocal code_lines, code_start
        if code_start is None:
            return
        text = "\n".join(code_lines)
        _emit("code_block", int(code_start), int(end_line), text)
        code_lines = []
        code_start = None

    i = 1
    while i <= len(lines):
        line = lines[i - 1]
        s = line.strip()

        if in_code:
            code_lines.append(line)
            if s.startswith("```"):
                in_code = False
                flush_code(i)
            i += 1
            continue

        if s.startswith("```"):
            flush_para(i - 1)
            in_code = True
            code_start = i
            code_lines = [line]
            i += 1
            continue

        hm = _heading_match(line)
        if hm is not None:
            flush_para(i - 1)
            level, title = hm
            section_stack = [(lv, t) for (lv, t) in section_stack if lv < level]
            section_stack.append((level, title))
            _emit("heading", i, i, title)
            i += 1
            continue

        if s == "":
            flush_para(i - 1)
            i += 1
            continue

        # List item (best-effort: one line + indented continuations).
        if RE_LIST_BULLET.match(line) or RE_LIST_NUM.match(line):
            flush_para(i - 1)
            start = i
            item_lines = [line]
            j = i + 1
            while j <= len(lines):
                nxt = lines[j - 1]
                if nxt.strip() == "":
                    break
                if _heading_match(nxt) is not None:
                    break
                if RE_LIST_BULLET.match(nxt) or RE_LIST_NUM.match(nxt):
                    break
                if len(nxt) - len(nxt.lstrip(" ")) <= 1:
                    break
                item_lines.append(nxt)
                j += 1
            _emit("list_item", start, j - 1, "\n".join(item_lines).rstrip())
            i = j
            continue

        # Table block (best-effort): consecutive |...| lines.
        if RE_TABLE_ROW.match(line):
            flush_para(i - 1)
            start = i
            tbl = [line]
            j = i + 1
            while j <= len(lines) and RE_TABLE_ROW.match(lines[j - 1]):
                tbl.append(lines[j - 1])
                j += 1
            _emit("table_block", start, j - 1, "\n".join(tbl).rstrip())
            i = j
            continue

        # Default: paragraph accumulation.
        if para_start is None:
            para_start = i
        para_lines.append(line)
        i += 1

    flush_para(len(lines))
    if in_code:
        in_code = False
        flush_code(len(lines))

    return atoms


@dataclass(frozen=True)
class Ref:
    path: str  # repo-relative posix path
    line: int


RE_REQ_TAG = re.compile(r"UVCC_REQ:\s*([A-Za-z0-9_\-.,]+)")
RE_GRP_TAG = re.compile(r"UVCC_REQ_GROUP:\s*([A-Za-z0-9_\-.,]+)")


def _iter_repo_files(repo_root: Path) -> Iterable[Path]:
    """
    Deterministic repo walk for tag scanning.
    """
    deny_dirs = {
        ".git",
        ".idea",
        ".vscode",
        ".cursor",
        "__pycache__",
        "node_modules",
        "out",
        "dist",
        "build",
        "target",
        ".venv",
        "venv",
        "lib",  # vendored deps in this repo
    }
    allow_ext = {
        ".py",
        ".md",
        ".txt",
        ".c",
        ".h",
        ".cpp",
        ".cc",
        ".cu",
        ".cuh",
        ".rs",
        ".go",
        ".java",
        ".kt",
        ".ts",
        ".tsx",
        ".js",
        ".sol",
        ".toml",
        ".yml",
        ".yaml",
        ".json",
    }
    for p in sorted(repo_root.rglob("*")):
        if not p.is_file():
            continue
        # quick path filter
        if any(part in deny_dirs for part in p.parts):
            continue
        if p.suffix.lower() not in allow_ext:
            continue
        yield p


def scan_uvcc_tags(repo_root: Path) -> Tuple[Dict[str, List[Ref]], Dict[str, List[Ref]]]:
    """
    Returns (req_id -> refs, group_id -> refs).
    """
    req: Dict[str, List[Ref]] = {}
    grp: Dict[str, List[Ref]] = {}

    for path in _iter_repo_files(repo_root):
        try:
            data = path.read_text(encoding="utf-8", errors="strict").splitlines()
        except Exception:
            continue
        rel = _relpath_if_possible(path, repo_root)
        for i, line in enumerate(data, start=1):
            m = RE_REQ_TAG.search(line)
            if m:
                ids = [s.strip() for s in m.group(1).split(",") if s.strip()]
                for rid in ids:
                    req.setdefault(rid, []).append(Ref(path=rel, line=i))
            m = RE_GRP_TAG.search(line)
            if m:
                ids = [s.strip() for s in m.group(1).split(",") if s.strip()]
                for gid in ids:
                    grp.setdefault(gid, []).append(Ref(path=rel, line=i))
    return req, grp


def _is_test_path(relpath: str) -> bool:
    rp = relpath.replace("\\", "/")
    if "/tests/" in rp or "/test/" in rp:
        return True
    if rp.endswith(".t.sol") or rp.endswith("_test.py") or rp.endswith("test.py"):
        return True
    return False


def _bucket_for_ref(relpath: str) -> str:
    rp = relpath.replace("\\", "/")
    if _is_test_path(rp):
        return "test"
    if rp.startswith("research/uvcc/uvcc-spec/"):
        return "spec"
    if rp.startswith("research/uvcc/uvcc-party/"):
        return "runtime"
    if rp.startswith("research/uvcc/uvcc-verifier/"):
        return "verifier"
    if rp.startswith("research/uvcc/uvcc-contracts/"):
        return "contracts"
    if rp.startswith("research/uvcc/uvcc-client/"):
        return "client"
    if rp.startswith("research/uvcc/uvcc-demo/"):
        return "demo"
    return "other"


def _format_refs(refs: Sequence[Ref], *, max_items: int = 6) -> str:
    if not refs:
        return ""
    rs = sorted(refs, key=lambda r: (r.path, int(r.line)))
    shown = rs[: int(max_items)]
    s = ", ".join(f"`{r.path}:{int(r.line)}`" for r in shown)
    if len(rs) > len(shown):
        s += f", …(+{len(rs) - len(shown)})"
    return s

def _md_escape(s: str) -> str:
    return s.replace("|", "\\|")


def atoms_to_json_bytes(*, doc_sha256: str, privacy_relpath: str, atoms: Sequence[Atom]) -> bytes:
    obj = {
        "schema": "uvcc.coverage.atoms.v1",
        "privacy_path": str(privacy_relpath),
        "privacy_sha256": str(doc_sha256),
        "atom_count": int(len(atoms)),
        "atoms": [
            {
                "idx": int(a.idx),
                "req_id": str(a.req_id),
                "group_id": str(a.group_id),
                "kind": str(a.kind),
                "start_line": int(a.start_line),
                "end_line": int(a.end_line),
                "section_path": list(a.section_path),
                "text": str(a.text),
            }
            for a in atoms
        ],
    }
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def generate_markdown_str(privacy_path: Path) -> Tuple[str, bytes, str]:
    lines = _read_lines(privacy_path)
    sha = _sha256_file(privacy_path)
    atoms = atomize_privacy_new(lines)

    repo_root = Path(__file__).resolve().parents[3]
    privacy_rel = _relpath_if_possible(privacy_path, repo_root)
    atoms_json = atoms_to_json_bytes(doc_sha256=sha, privacy_relpath=privacy_rel, atoms=atoms)
    req_refs, grp_refs = scan_uvcc_tags(repo_root)
    md: List[str] = []
    md.append("# UVCC Coverage Matrix — `research/privacy_new.txt`")
    md.append("")
    md.append("This file is auto-generated from the latest `research/privacy_new.txt` and is used as the **100% coverage gate**.")
    md.append("")
    md.append(f"- **Source**: `{privacy_rel}`")
    md.append(f"- **SHA256**: `{sha}`")
    md.append(f"- **Total lines**: `{len(lines)}`")
    md.append(f"- **Total atoms**: `{len(atoms)}`")
    md.append("")
    md.append("## Atom coverage")
    md.append("")
    md.append("| GroupID | ReqID | Kind | LineRange | SectionPath | Text | SpecRefs | RuntimeRefs | VerifierRefs | ContractsRefs | ClientRefs | DemoRefs | TestRefs |")
    md.append("|---|---|---|:---:|---|---|---|---|---|---|---|---|---|")
    for a in atoms:
        text_one = _collapse_ws(a.text)
        if len(text_one) > 180:
            text_one = text_one[:177] + "..."

        refs_all: List[Ref] = []
        refs_all.extend(grp_refs.get(a.group_id, []))
        refs_all.extend(req_refs.get(a.req_id, []))
        buckets: Dict[str, List[Ref]] = {k: [] for k in ["spec", "runtime", "verifier", "contracts", "client", "demo", "test"]}
        for r in refs_all:
            b = _bucket_for_ref(r.path)
            if b in buckets:
                buckets[b].append(r)
        md.append(
            "| "
            + " | ".join(
                [
                    _md_escape(a.group_id),
                    _md_escape(a.req_id),
                    _md_escape(a.kind),
                    f"`{a.start_line}:{a.end_line}`",
                    _md_escape(" > ".join(a.section_path) if a.section_path else "(root)"),
                    _md_escape(text_one),
                    _format_refs(buckets["spec"]),
                    _format_refs(buckets["runtime"]),
                    _format_refs(buckets["verifier"]),
                    _format_refs(buckets["contracts"]),
                    _format_refs(buckets["client"]),
                    _format_refs(buckets["demo"]),
                    _format_refs(buckets["test"]),
                ]
            )
            + " |"
        )
    md.append("")
    md.append("## Notes")
    md.append("")
    md.append("- The atom list is also written to `privacy_new_atoms.json` (machine source of truth).")
    md.append("- `tag-scan-gate` will populate Spec/Runtime/Verifier/Contracts/Client/Demo/Test refs via `UVCC_REQ:` tags.")
    md.append("- `--check` fails if regeneration would change outputs.")
    md.append("")
    return sha, atoms_json, "\n".join(md)


def write_outputs(*, privacy_path: Path, out_path: Path, atoms_out: Path) -> None:
    sha, atoms_json, md = generate_markdown_str(privacy_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atoms_out.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    atoms_out.write_bytes(atoms_json)


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--privacy",
        default=str(Path("research/privacy_new.txt")),
        help="Path to privacy_new.txt",
    )
    ap.add_argument(
        "--out",
        default=str(Path("research/uvcc/uvcc-spec/coverage/privacy_new_coverage_matrix.md")),
        help="Output markdown path",
    )
    ap.add_argument(
        "--atoms-out",
        default=str(Path("research/uvcc/uvcc-spec/coverage/privacy_new_atoms.json")),
        help="Output atoms JSON path",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="Check that the generated outputs match (--out, --atoms-out) (CI gate).",
    )
    ap.add_argument(
        "--gate",
        action="store_true",
        help="Fail if any atom lacks BOTH (implementation refs) and (test refs) via UVCC_REQ/UVCC_REQ_GROUP tags.",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    privacy_path = Path(args.privacy)
    out_path = Path(args.out)
    atoms_out = Path(args.atoms_out)
    if not privacy_path.exists():
        raise SystemExit(f"privacy file not found: {privacy_path}")

    if args.check:
        _, expected_atoms, expected_md = generate_markdown_str(privacy_path)
        got_md = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
        if got_md != expected_md:
            raise SystemExit(
                f"coverage matrix out of date: regenerate with `{Path(__file__).as_posix()} --privacy {privacy_path} --out {out_path}`"
            )
        got_atoms = atoms_out.read_bytes() if atoms_out.exists() else b""
        if got_atoms != expected_atoms:
            raise SystemExit(
                f"atoms.json out of date: regenerate with `{Path(__file__).as_posix()} --privacy {privacy_path} --atoms-out {atoms_out}`"
            )
        return 0

    write_outputs(privacy_path=privacy_path, out_path=out_path, atoms_out=atoms_out)
    if args.gate:
        repo_root = Path(__file__).resolve().parents[3]
        req_refs, grp_refs = scan_uvcc_tags(repo_root)
        atoms = atomize_privacy_new(_read_lines(privacy_path))

        missing_impl = 0
        missing_test = 0
        miss_groups_impl: Dict[str, int] = {}
        miss_groups_test: Dict[str, int] = {}

        for a in atoms:
            refs_all: List[Ref] = []
            refs_all.extend(grp_refs.get(a.group_id, []))
            refs_all.extend(req_refs.get(a.req_id, []))
            impl_ok = any(_bucket_for_ref(r.path) in {"spec", "runtime", "verifier", "contracts", "client", "demo"} for r in refs_all)
            test_ok = any(_bucket_for_ref(r.path) == "test" for r in refs_all)
            if not impl_ok:
                missing_impl += 1
                miss_groups_impl[a.group_id] = miss_groups_impl.get(a.group_id, 0) + 1
            if not test_ok:
                missing_test += 1
                miss_groups_test[a.group_id] = miss_groups_test.get(a.group_id, 0) + 1

        if missing_impl or missing_test:
            # Show a small deterministic summary to guide the next steps.
            top_impl = sorted(miss_groups_impl.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
            top_test = sorted(miss_groups_test.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
            msg = [
                f"UVCC coverage gate failed: atoms={len(atoms)} missing_impl={missing_impl} missing_test={missing_test}",
                "TopMissingImplGroups=" + ",".join(f"{g}:{n}" for g, n in top_impl),
                "TopMissingTestGroups=" + ",".join(f"{g}:{n}" for g, n in top_test),
            ]
            raise SystemExit("\n".join(msg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


