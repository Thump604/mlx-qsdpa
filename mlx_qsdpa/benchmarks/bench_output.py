"""Output formatting helpers for mlx-qsdpa benchmark CLIs."""


def format_config(H_q, H_kv, bits, gs):
    gqa = "non-gqa" if H_q == H_kv else f"gqa-{H_q // H_kv}x"
    return f"{gqa} {bits}b gs{gs}"


def print_header():
    print(f"{'config':<20} {'path':<14} {'N':>7} {'qL':>4} "
          f"{'median':>9} {'p5':>9} {'p95':>9} {'GB/s':>8} {'%peak':>6}")
    print("-" * 96)


def print_row(r):
    cfg = format_config(r["H_q"], r["H_kv"], r["bits"], r["group_size"])
    print(f"{cfg:<20} {r['path']:<14} {r['N']:>7} {r['qL']:>4} "
          f"{r['median_us']:>8.1f}u {r['p5_us']:>8.1f}u {r['p95_us']:>8.1f}u "
          f"{r['achieved_gbps']:>7.1f} {r['pct_peak']:>5.1f}%")
