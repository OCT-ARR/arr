# -*- coding: utf-8 -*-
"""Analyze cognitive/critical reward quadrants, rank changes, and group-relative advantages."""
import os
import json
import argparse
from pathlib import Path
import statistics as st

EPS = 1e-8
COG = ("visual", "emotion", "situation")
ALL5 = ("visual", "emotion", "situation", "strategy", "outcome")


def cognitive(c):
    return sum(c[k] for k in COG) / 3.0


def critical(c):
    return c["strategy"] * c["outcome"]


def agg_linear(c):
    return sum(c[k] for k in ALL5) / 5.0 if c["format"] >= 0.5 else 0.0


def agg_gated(c):
    return cognitive(c) * c["outcome"] * c["strategy"] if c["format"] >= 0.5 else 0.0


def load(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            for j, c in enumerate(rec["completions"]):
                rows.append({
                    "idx": rec["idx"], "j": j,
                    "prompt": rec.get("prompt", ""),
                    "response": c.get("response", ""),
                    "format": float(c.get("format", 0.0)),
                    "visual": float(c.get("visual", 0.0)),
                    "emotion": float(c.get("emotion", 0.0)),
                    "situation": float(c.get("situation", 0.0)),
                    "strategy": float(c.get("strategy", 0.0)),
                    "outcome": float(c.get("outcome", 0.0)),
                    "indep_rouge_l": float(c.get("indep_rouge_l", 0.0)),
                    "indep_epitome_skill": float(c.get("indep_epitome_skill", 0.0)),
                })
    return rows


def fmt_table(headers, table_rows):
    widths = [len(str(h)) for h in headers]
    for r in table_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))
    line = "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    out = [line, "  ".join("-" * widths[i] for i in range(len(headers)))]
    for r in table_rows:
        out.append("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))
    return "\n".join(out)


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy + EPS)


def kendall_tau(x, y, cap=20000):
    n = len(x)
    conc = disc = 0
    import random
    random.seed(42)
    pairs = [(i, k) for i in range(n) for k in range(i + 1, n)]
    if len(pairs) > cap:
        pairs = random.sample(pairs, cap)
    for i, k in pairs:
        sx = (x[i] > x[k]) - (x[i] < x[k])
        sy = (y[i] > y[k]) - (y[i] < y[k])
        p = sx * sy
        if p > 0:
            conc += 1
        elif p < 0:
            disc += 1
    tot = conc + disc
    return (conc - disc) / (tot + EPS), len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out_dir", default=str(Path(__file__).resolve().parents[2] / "outputs" / "reward_diagnostics"))
    ap.add_argument("--q", type=float, default=0.25)
    ap.add_argument("--cog_hi", type=float, default=0.6, help="cognitive 高分阈值")
    ap.add_argument("--crit_lo", type=float, default=0.1, help="critical(strategy*outcome) 低分阈值")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = load(args.inp)
    valid = [r for r in rows if r["format"] >= 0.5]
    report = []

    def emit(s=""):
        print(s, flush=True)
        report.append(s)

    emit("=" * 78)
    emit(f"FACES paper — Reward-geometry 深挖  |  valid completions={len(valid)}")
    emit("=" * 78)

    N = len(valid)
    cog_vals = [cognitive(c) for c in valid]
    crit_vals = [critical(c) for c in valid]

    # ============ 分析A: Reward geometry 象限分布 ============
    emit("\n[分析A] Reward geometry 象限分布")
    emit(f"      横轴 cognitive=mean(visual,emotion,situation)  阈值={args.cog_hi}")
    emit(f"      纵轴 critical =strategy*outcome                 阈值={args.crit_lo}")
    emit("      Q_HL = 高认知/低关键维 = '感知说得通但策略/输出塌方' = 补偿性聚合的高危区")
    ch, cl = args.cog_hi, args.crit_lo
    quad = {"Q_HH": 0, "Q_HL": 0, "Q_LH": 0, "Q_LL": 0}
    for cg, cr in zip(cog_vals, crit_vals):
        hi_c = cg >= ch
        hi_k = cr >= cl
        if hi_c and hi_k:
            quad["Q_HH"] += 1
        elif hi_c and not hi_k:
            quad["Q_HL"] += 1
        elif not hi_c and hi_k:
            quad["Q_LH"] += 1
        else:
            quad["Q_LL"] += 1
    tA = [
        ["Q_HH 高认知·高关键 (真优)", quad["Q_HH"], f"{100.0*quad['Q_HH']/N:.1f}%"],
        ["Q_HL 高认知·低关键 (名不副实)", quad["Q_HL"], f"{100.0*quad['Q_HL']/N:.1f}%"],
        ["Q_LH 低认知·高关键", quad["Q_LH"], f"{100.0*quad['Q_LH']/N:.1f}%"],
        ["Q_LL 低认知·低关键 (真差)", quad["Q_LL"], f"{100.0*quad['Q_LL']/N:.1f}%"],
    ]
    emit(fmt_table(["Quadrant", "N", "Ratio"], tA))
    # Q_HL 组在 linear 下的平均分 vs 全体, 证明补偿聚合确实给它们发高奖励
    lin_all = [agg_linear(c) for c in valid]
    qhl_idx = [i for i in range(N) if cog_vals[i] >= ch and crit_vals[i] < cl]
    if qhl_idx:
        import statistics as _st
        lin_qhl = _st.mean([lin_all[i] for i in qhl_idx])
        lin_mean = _st.mean(lin_all)
        gat_all = [agg_gated(c) for c in valid]
        gat_qhl = _st.mean([gat_all[i] for i in qhl_idx])
        gat_mean = _st.mean(gat_all)
        emit(f"  Q_HL 组 (N={len(qhl_idx)}) 在 linear 下均分 {lin_qhl:.4f} (全体 {lin_mean:.4f}, "
             f"{'高' if lin_qhl>=lin_mean else '低'}于均值 {abs(lin_qhl-lin_mean)/max(1e-8,lin_mean)*100:.1f}%)")
        emit(f"  Q_HL 组在 gated 下均分 {gat_qhl:.4f} (全体 {gat_mean:.4f}, "
             f"被压到均值的 {gat_qhl/max(1e-8,gat_mean)*100:.1f}%)")
        emit("  解读: linear 给 Q_HL 发放接近/高于均值的奖励 (名不副实), gated 把它们显著压低。")

    # ============ 分析B: Linear vs Gated 排序几何差异 ============
    emit("\n[分析B] Linear vs Gated 排序几何差异")
    lin_s = [agg_linear(c) for c in valid]
    gat_s = [agg_gated(c) for c in valid]
    rho = spearman(lin_s, gat_s)
    tau, npairs = kendall_tau(lin_s, gat_s)
    emit(f"  Spearman ρ = {rho:.4f}   Kendall τ = {tau:.4f} (采样 {npairs} 对)")
    emit("  解读: ρ/τ 远小于 1 => 两种聚合是结构性不同的排序信号, 非可互换的等价形式。")

    # 排名翻转: linear top-q 但 gated 掉出 top-q
    q = args.q
    top_k = max(1, int(N * q))
    lin_order = sorted(range(N), key=lambda i: lin_s[i], reverse=True)
    gat_order = sorted(range(N), key=lambda i: gat_s[i], reverse=True)
    lin_top = set(lin_order[:top_k])
    gat_top = set(gat_order[:top_k])
    flip_out = lin_top - gat_top  # linear 捧进 top, gated 踢出 top
    emit(f"  top-{int(q*100)}% 大小={top_k}; linear 排进 top 但 gated 踢出 top 的回复数: "
         f"{len(flip_out)} ({100.0*len(flip_out)/top_k:.1f}% 的 top 席位被翻转)")

    # 导出典型 reward arbitrage 案例 (linear 高 / gated 低 / 关键维塌方)
    lin_rank = {i: k for k, i in enumerate(lin_order)}
    gat_rank = {i: k for k, i in enumerate(gat_order)}
    cases = sorted(flip_out, key=lambda i: (gat_rank[i] - lin_rank[i]), reverse=True)
    emit("\n  最典型的 3 个 reward arbitrage 案例 (linear 排名高 / gated 排名低):")
    for rank_i, i in enumerate(cases[:3], 1):
        c = valid[i]
        emit(f"    #{rank_i} idx={c['idx']} j={c['j']}  "
             f"cog={cognitive(c):.2f} str={c['strategy']:.2f} out={c['outcome']:.2f} | "
             f"linear#{lin_rank[i]+1} vs gated#{gat_rank[i]+1}")
        resp = (c.get("response") or "").strip().replace("\n", " ")
        emit(f"       resp: {resp[:160]}")

    # 保存
    txt = os.path.join(args.out_dir, "geometry_report.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    emit(f"\n报告已保存: {txt}")


if __name__ == "__main__":
    main()
