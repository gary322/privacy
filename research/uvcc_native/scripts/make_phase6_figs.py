#!/usr/bin/env python3
import argparse
import os
import re
from collections import defaultdict, Counter

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np


RE_NCCL_INIT = re.compile(
    r"Init timings - ncclCommInitRank: rank\s+\d+\s+nranks\s+(\d+)\s+total\s+([0-9]*\.[0-9]+|[0-9]+)"
)

RE_META_TOPO = re.compile(r"R=(\d+)\s+S=(\d+)\s+T=(\d+)\s+M=(\d+)")
RE_GID = re.compile(r"phase6_init_group gid=([^\s]+)")
RE_COORD_P = re.compile(r"-p(\d+)-")
RE_COORD_R = re.compile(r"-r(\d+)-")
RE_COORD_S = re.compile(r"-s(\d+)-")
RE_COORD_T = re.compile(r"-t(\d+)-")

def parse_logs(path: str):
    counts = Counter()
    nccl_totals_by_nranks = defaultdict(list)
    nccl_totals_by_group = {"tp": [], "pp": [], "dp": []}
    dp_nccl_totals_by_st = defaultdict(list)  # (s,t) -> [total_seconds]

    re_epoch = re.compile(r"epoch_root=0x[0-9a-f]+")
    re_dp_ready = re.compile(r"\bphase6_dp_ready\b")
    re_dp_ok = re.compile(r"\bphase6_dp_after_bwd_ok\b")
    re_fwd_open_ok = re.compile(r"\bphase6_fwd_open_ok mb=\d+\b")
    re_bwd_open_ok = re.compile(r"\bphase6_bwd_open_ok mb=\d+\b")
    re_uid_wait = re.compile(r"\bphase6_uid_wait\b")
    # Important: count *specific* failure signatures, not any mention of the word "timeout"
    # (the explained log includes runner stack traces that mention internal timeout helpers).
    re_worker_timeout = re.compile(r"(timeout waiting for|Timeout was reached)", re.IGNORECASE)
    re_transport_abort = re.compile(r"(retransmit tries_max exceeded|received NACK)", re.IGNORECASE)
    re_runner_ssh_banner = re.compile(r"Exception \(client\): Error reading SSH protocol banner", re.IGNORECASE)

    # Per-(s,t) and per-(r,s,t) aggregations
    dp_ready_by_st = defaultdict(int)
    fwd_ok_by_s = defaultdict(int)
    bwd_ok_by_s = defaultdict(int)
    fwd_ok_by_st = defaultdict(int)
    bwd_ok_by_st = defaultdict(int)
    uid_wait_by_group = Counter()
    errors_by_group = Counter()
    epoch_root_by_rst = defaultdict(int)

    # Track topology for grids
    topo = {"R": None, "S": None, "T": None, "M": None}
    parties_seen = set()

    # "Context" derived from last seen group (safe within a worker chunk)
    ctx = {"group": None, "p": None, "r": None, "s": None, "t": None}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Topology
            mtop = RE_META_TOPO.search(line)
            if mtop:
                topo["R"], topo["S"], topo["T"], topo["M"] = map(int, mtop.groups())

            # Group context
            mgid = RE_GID.search(line)
            if mgid:
                gid = mgid.group(1)
                group = "tp" if "-nccl-tp-" in gid else ("pp" if "-nccl-pp-" in gid else ("dp" if "-nccl-dp-" in gid else None))
                p = RE_COORD_P.search(gid)
                r = RE_COORD_R.search(gid)
                s = RE_COORD_S.search(gid)
                t = RE_COORD_T.search(gid)
                ctx["group"] = group
                ctx["p"] = int(p.group(1)) if p else None
                ctx["r"] = int(r.group(1)) if r else None
                ctx["s"] = int(s.group(1)) if s else None
                ctx["t"] = int(t.group(1)) if t else None
                if ctx["p"] is not None:
                    parties_seen.add(ctx["p"])

            if re_epoch.search(line):
                counts["epoch_root"] += 1
                if ctx["r"] is not None and ctx["s"] is not None and ctx["t"] is not None:
                    epoch_root_by_rst[(ctx["r"], ctx["s"], ctx["t"])] += 1
            if re_dp_ready.search(line):
                counts["dp_ready"] += 1
                if ctx["s"] is not None and ctx["t"] is not None:
                    dp_ready_by_st[(ctx["s"], ctx["t"])] += 1
            if re_dp_ok.search(line):
                counts["dp_ok"] += 1
            if re_fwd_open_ok.search(line):
                counts["fwd_open_ok"] += 1
                if ctx["s"] is not None:
                    fwd_ok_by_s[ctx["s"]] += 1
                if ctx["s"] is not None and ctx["t"] is not None:
                    fwd_ok_by_st[(ctx["s"], ctx["t"])] += 1
            if re_bwd_open_ok.search(line):
                counts["bwd_open_ok"] += 1
                if ctx["s"] is not None:
                    bwd_ok_by_s[ctx["s"]] += 1
                if ctx["s"] is not None and ctx["t"] is not None:
                    bwd_ok_by_st[(ctx["s"], ctx["t"])] += 1
            if re_uid_wait.search(line):
                counts["uid_wait"] += 1
                if ctx["group"]:
                    uid_wait_by_group[ctx["group"]] += 1

            # Robustness/error indicators (scoped to meaningful patterns)
            if re_runner_ssh_banner.search(line):
                counts["runner_ssh_banner_errors"] += 1
            if re_worker_timeout.search(line):
                counts["worker_timeouts"] += 1
                if ctx["group"]:
                    errors_by_group[(ctx["group"], "timeout")] += 1
            if re_transport_abort.search(line):
                counts["transport_aborts"] += 1
                if ctx["group"]:
                    errors_by_group[(ctx["group"], "retransmit")] += 1

            m = RE_NCCL_INIT.search(line)
            if m:
                nranks = int(m.group(1))
                total = float(m.group(2))
                nccl_totals_by_nranks[nranks].append(total)
                if ctx["group"] in ("tp", "pp", "dp"):
                    nccl_totals_by_group[ctx["group"]].append(total)
                if ctx["group"] == "dp" and ctx["s"] is not None and ctx["t"] is not None:
                    dp_nccl_totals_by_st[(ctx["s"], ctx["t"])].append(total)

    return {
        "counts": counts,
        "nccl_by_nranks": nccl_totals_by_nranks,
        "nccl_by_group": nccl_totals_by_group,
        "dp_ready_by_st": dp_ready_by_st,
        "dp_nccl_totals_by_st": dp_nccl_totals_by_st,
        "fwd_ok_by_s": fwd_ok_by_s,
        "bwd_ok_by_s": bwd_ok_by_s,
        "fwd_ok_by_st": fwd_ok_by_st,
        "bwd_ok_by_st": bwd_ok_by_st,
        "uid_wait_by_group": uid_wait_by_group,
        "errors_by_group": errors_by_group,
        "epoch_root_by_rst": epoch_root_by_rst,
        "topo": topo,
        "parties_seen": parties_seen,
    }


def plot_nccl_init_hist(nccl_totals_by_nranks, out_path):
    plt.figure(figsize=(8, 4.5))
    labels = []
    data = []
    colors = []
    # Bucket order deterministic: TP=2, PP=4, DP=8 (when present)
    for nranks, label, color in [(2, "TP (nranks=2)", "#1f77b4"),
                                 (4, "PP (nranks=4)", "#2ca02c"),
                                 (8, "DP (nranks=8)", "#d62728")]:
        if nranks in nccl_totals_by_nranks and nccl_totals_by_nranks[nranks]:
            labels.append(label)
            data.append(nccl_totals_by_nranks[nranks])
            colors.append(color)
    if not data:
        # nothing to plot; create a blank canvas
        plt.text(0.5, 0.5, "No NCCL init timing lines found", ha="center", va="center")
    else:
        plt.hist(data, bins=30, label=labels, color=colors, alpha=0.7, edgecolor="black", linewidth=0.3)
        plt.legend()
        plt.xlabel("ncclCommInitRank total time (s)")
        plt.ylabel("count")
        plt.title("NCCL initialization timings distribution")
        plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_event_counts(counts, out_path):
    keys = ["epoch_root", "dp_ready", "dp_ok", "fwd_open_ok", "bwd_open_ok", "uid_wait"]
    labels = ["epoch_root", "DP ready", "DP ok", "FWD OPEN ok", "BWD OPEN ok", "UID waits"]
    values = [counts.get(k, 0) for k in keys]

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(labels, values, color="#4c72b0")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("count")
    plt.title("Phase 6 event counts (final run)")
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v}", ha="center", va="bottom", fontsize=8)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_error_counts(counts, out_path):
    keys = ["worker_timeouts", "transport_aborts", "runner_ssh_banner_errors"]
    labels = ["Worker timeouts", "Transport aborts", "Runner SSH banner errors"]
    values = [counts.get(k, 0) for k in keys]

    plt.figure(figsize=(9, 4))
    bars = plt.bar(labels, values, color="#dd8452")
    plt.ylabel("count")
    plt.title("Robustness indicators (final run)")
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v}", ha="center", va="bottom", fontsize=9)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_nccl_cdf(nccl_by_group, out_path):
    plt.figure(figsize=(8, 4.5))
    for group, color, label in [("tp", "#1f77b4", "TP"), ("pp", "#2ca02c", "PP"), ("dp", "#d62728", "DP")]:
        vals = sorted(nccl_by_group.get(group, []))
        if not vals:
            continue
        y = np.linspace(0, 1, num=len(vals), endpoint=True)
        plt.step(vals, y, where="post", label=label, color=color)
    if not plt.gca().has_data():
        plt.text(0.5, 0.5, "No NCCL measurements", ha="center", va="center")
    else:
        plt.xlabel("ncclCommInitRank total time (s)")
        plt.ylabel("CDF")
        plt.title("NCCL initialization time CDF by group")
        plt.grid(True, alpha=0.25)
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_fwd_bwd_by_stage(fwd_by_s, bwd_by_s, S, out_path):
    xs = list(range(S))
    fwd_vals = [fwd_by_s.get(s, 0) for s in xs]
    bwd_vals = [bwd_by_s.get(s, 0) for s in xs]
    width = 0.38
    plt.figure(figsize=(8, 4.5))
    plt.bar([x - width/2 for x in xs], fwd_vals, width, label="FWD OPEN ok", color="#4c72b0")
    plt.bar([x + width/2 for x in xs], bwd_vals, width, label="BWD OPEN ok", color="#55a868")
    plt.xlabel("stage s")
    plt.ylabel("count")
    plt.title("OPEN completions by pipeline stage")
    plt.xticks(xs, [str(s) for s in xs])
    plt.legend()
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_dp_ready_heatmap(dp_ready_by_st, S, T, out_path, expected_per_cell=None):
    mat = np.zeros((S, T), dtype=float)
    for (s, t), v in dp_ready_by_st.items():
        if 0 <= s < S and 0 <= t < T:
            mat[s, t] = v
    plt.figure(figsize=(6, 4.5))
    if expected_per_cell is not None and expected_per_cell > 0:
        show = mat / float(expected_per_cell)
        im = plt.imshow(show, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
        title = f"DP ready fraction per (s,t), expected={expected_per_cell} per cell"
        # annotate with counts
        for s in range(S):
            for t in range(T):
                v = int(mat[s, t])
                frac = show[s, t]
                color = "white" if frac > 0.6 else "black"
                plt.text(t, s, f"{v}/{expected_per_cell}", ha="center", va="center", color=color, fontsize=10)
    else:
        im = plt.imshow(mat, aspect="auto", cmap="Blues")
        title = "DP ready counts per (s,t)"
        for s in range(S):
            for t in range(T):
                v = int(mat[s, t])
                plt.text(t, s, f"{v}", ha="center", va="center", color="black", fontsize=10)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("t (tensor rank)")
    plt.ylabel("s (stage)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_dp_init_p50_heatmap(dp_nccl_totals_by_st, S, T, out_path):
    mat = np.full((S, T), np.nan, dtype=float)
    for (s, t), vals in dp_nccl_totals_by_st.items():
        if 0 <= s < S and 0 <= t < T and vals:
            vs = sorted(vals)
            mat[s, t] = vs[len(vs)//2]  # p50
    plt.figure(figsize=(6, 4.5))
    # If all NaN, render message
    if np.all(np.isnan(mat)):
        plt.text(0.5, 0.5, "No DP NCCL init timing samples", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
        return

    im = plt.imshow(mat, aspect="auto", cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("t (tensor rank)")
    plt.ylabel("s (stage)")
    plt.title("DP NCCL init time p50 (s) per (s,t)")
    # annotate
    for s in range(S):
        for t in range(T):
            if not np.isnan(mat[s, t]):
                plt.text(t, s, f"{mat[s,t]:.1f}", ha="center", va="center", color="white", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_errors_by_group(errors_by_group, out_path):
    labels = []
    vals = []
    for group in ("tp", "pp", "dp"):
        for etype in ("timeout", "retransmit"):
            labels.append(f"{group}:{etype}")
            vals.append(errors_by_group.get((group, etype), 0))
    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, vals, color="#c44e52")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("count")
    plt.title("Errors by group/type")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v}", ha="center", va="bottom", fontsize=8)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_open_heatmap(counts_by_st, S, T, out_path, title):
    mat = np.zeros((S, T), dtype=float)
    for (s, t), v in counts_by_st.items():
        if 0 <= s < S and 0 <= t < T:
            mat[s, t] = v
    plt.figure(figsize=(6, 4.5))
    im = plt.imshow(mat, aspect="auto", cmap="Greens")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("t (tensor rank)")
    plt.ylabel("s (stage)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_epoch_roots_heatmap(epoch_root_by_rst, S, T, out_path):
    mat = np.zeros((S, T), dtype=float)
    for (r, s, t), v in epoch_root_by_rst.items():
        if 0 <= s < S and 0 <= t < T:
            mat[s, t] += v  # sum over replicas
    plt.figure(figsize=(6, 4.5))
    im = plt.imshow(mat, aspect="auto", cmap="Purples")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("t (tensor rank)")
    plt.ylabel("s (stage)")
    plt.title("Epoch root emissions (sum over replicas) per (s,t)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def write_metrics_table_tex(out_path, counts, parsed):
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Key Phase 6 metrics (derived from consolidated logs)}")
    lines.append("\\label{tab:metrics}")
    lines.append("\\begin{tabular}{lrr}")
    lines.append("\\toprule")
    lines.append("Metric & Count/Value \\\\")
    lines.append("\\midrule")
    for k in ["epoch_root", "dp_ready", "dp_ok", "fwd_open_ok", "bwd_open_ok", "uid_wait",
              "worker_timeouts", "transport_aborts", "runner_ssh_banner_errors"]:
        lines.append(f"{k.replace('_',' ')} & {counts.get(k,0)} \\\\")
    # NCCL summaries
    for group in ("tp", "pp", "dp"):
        vals = sorted(parsed['nccl_by_group'].get(group, []))
        if vals:
            p50 = vals[len(vals)//2]
            lines.append(
                f"nccl init {group} (n={len(vals)}) min/p50/max & "
                f"{min(vals):.3f}/{p50:.3f}/{max(vals):.3f} s \\\\"
            )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    with open(out_path, "w") as fw:
        fw.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explained", required=True, help="Path to all_logs_explained.md")
    ap.add_argument("--outdir", required=True, help="Directory to write figures to")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    parsed = parse_logs(args.explained)
    counts = parsed["counts"]
    topo = parsed["topo"]

    plot_nccl_init_hist(parsed["nccl_by_nranks"], os.path.join(args.outdir, "nccl_init_hist.png"))
    plot_nccl_cdf(parsed["nccl_by_group"], os.path.join(args.outdir, "nccl_init_cdf.png"))
    plot_event_counts(counts, os.path.join(args.outdir, "event_counts.png"))
    plot_error_counts(counts, os.path.join(args.outdir, "error_counts.png"))
    if topo["S"] is not None and topo["T"] is not None:
        if parsed["fwd_ok_by_s"] or parsed["bwd_ok_by_s"]:
            plot_fwd_bwd_by_stage(parsed["fwd_ok_by_s"], parsed["bwd_ok_by_s"], topo["S"], os.path.join(args.outdir, "fwd_bwd_by_stage.png"))
        # Expected per (s,t) cell: R replicas per party times number of parties observed.
        parties = parsed.get("parties_seen") or set()
        n_parties = len(parties) if parties else 3
        expected = (topo["R"] or 0) * n_parties if topo["R"] is not None else None
        plot_dp_ready_heatmap(parsed["dp_ready_by_st"], topo["S"], topo["T"], os.path.join(args.outdir, "dp_ready_heatmap.png"), expected_per_cell=expected)
        plot_dp_init_p50_heatmap(parsed["dp_nccl_totals_by_st"], topo["S"], topo["T"], os.path.join(args.outdir, "dp_init_p50_heatmap.png"))
    plot_errors_by_group(parsed["errors_by_group"], os.path.join(args.outdir, "errors_by_group.png"))
    if topo["S"] is not None and topo["T"] is not None:
        if parsed["fwd_ok_by_st"]:
            plot_open_heatmap(parsed["fwd_ok_by_st"], topo["S"], topo["T"], os.path.join(args.outdir, "fwd_ok_heatmap.png"),
                              "FWD OPEN ok per (s,t)")
        if parsed["bwd_ok_by_st"]:
            plot_open_heatmap(parsed["bwd_ok_by_st"], topo["S"], topo["T"], os.path.join(args.outdir, "bwd_ok_heatmap.png"),
                              "BWD OPEN ok per (s,t)")
        if parsed["epoch_root_by_rst"]:
            plot_epoch_roots_heatmap(parsed["epoch_root_by_rst"], topo["S"], topo["T"], os.path.join(args.outdir, "epoch_roots_heatmap.png"))
    write_metrics_table_tex(os.path.join(args.outdir, "metrics_table.tex"), counts, parsed)

    # Write a brief metrics summary alongside figures
    with open(os.path.join(args.outdir, "metrics_summary.txt"), "w") as fw:
        fw.write("Phase 6 metrics (parsed from explained logs)\n")
        for k in sorted(counts.keys()):
            fw.write(f"{k}: {counts[k]}\n")
        for nranks in sorted(parsed["nccl_by_nranks"].keys()):
            vals = parsed["nccl_by_nranks"][nranks]
            if vals:
                fw.write(f"nccl_init_totals[nranks={nranks}]: n={len(vals)} "
                         f"min={min(vals):.3f} p50={sorted(vals)[len(vals)//2]:.3f} "
                         f"max={max(vals):.3f}\n")
        for group in ("tp", "pp", "dp"):
            vals = parsed["nccl_by_group"].get(group, [])
            if vals:
                fw.write(f"nccl_init_totals[group={group}]: n={len(vals)} "
                         f"min={min(vals):.3f} p50={sorted(vals)[len(vals)//2]:.3f} "
                         f"max={max(vals):.3f}\n")
        fw.write(f"topology: R={topo['R']} S={topo['S']} T={topo['T']} M={topo['M']}\n")

if __name__ == "__main__":
    main()


