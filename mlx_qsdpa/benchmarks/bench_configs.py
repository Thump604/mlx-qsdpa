"""Benchmark configuration sweep helpers."""

DECODE_SEQ_LENS = [1024, 4096, 16384, 32768, 65536, 131072]
PREFILL_SEQ_LENS = [4096, 16384, 65536]
PREFILL_QL = [128, 512]
HEAD_CONFIGS = [(2, 2), (32, 2)]  # (H_q, H_kv): non-GQA, GQA-16x
GROUP_SIZES_HEADLINE = [32]
GROUP_SIZES_ALL = [32, 64, 128]
D = 256
B = 1


def _parse_seq_filter(value):
    return {int(x) for x in value.split(",")} if value else None


def _parse_head_configs(value):
    if not value:
        return HEAD_CONFIGS
    hq, hkv = (int(part) for part in value.split(",", 1))
    return [(hq, hkv)]


def _decode_configs(head_cfgs, seq_filter, group_sizes):
    for hq, hkv in head_cfgs:
        for gs in group_sizes:
            for seq_len in DECODE_SEQ_LENS:
                if seq_filter is None or seq_len in seq_filter:
                    yield (B, hq, hkv, seq_len, D, 1, 4, gs)


def _prefill_configs(head_cfgs, seq_filter):
    for hq, hkv in head_cfgs:
        for query_len in PREFILL_QL:
            for seq_len in PREFILL_SEQ_LENS:
                if seq_filter is None or seq_len in seq_filter:
                    yield (B, hq, hkv, seq_len, D, query_len, 4, 32)


def build_configs(args):
    """Build list of (B, H_q, H_kv, N, D, qL, bits, gs) from CLI args."""
    seq_filter = _parse_seq_filter(args.seq_len)
    head_cfgs = _parse_head_configs(args.heads)
    group_sizes = GROUP_SIZES_HEADLINE if args.headline_only else GROUP_SIZES_ALL

    configs = []
    if not args.prefill_only:
        configs.extend(_decode_configs(head_cfgs, seq_filter, group_sizes))
    if not args.decode_only:
        configs.extend(_prefill_configs(head_cfgs, seq_filter))
    return configs
