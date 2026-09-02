from __future__ import absolute_import, division, print_function

"""Evaluate coded protection of physical prefixes in W1 BERT models.

The script loads a previously permuted model and its physical-prefix mask. Each
protected sign-bit stream is divided into 16 x 16 tiles and encoded using local
row/column parity constraints plus deterministic degree-4 algebraic checks.
Both data and parity bits pass through a binary symmetric channel, after which
noisy belief propagation repairs the protected prefix. Unprotected weights
remain subject to the same sign-bit error rate.

"""

import argparse
import copy
import gc
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler


NUM_LAYERS = 12
TYPES = ("q", "k", "v", "o", "f1", "f2")
ROW_PREFIX_TYPES = frozenset(("q", "k", "v", "f1"))
TILE_SIZE = 16
SEGMENT_LENGTH = 4
HASH_DEGREE = 4
TILE_BITS = TILE_SIZE * TILE_SIZE
QKVO_SUFFIXES = {
    "q": "attention.self.query.weight",
    "k": "attention.self.key.weight",
    "v": "attention.self.value.weight",
    "o": "attention.output.dense.weight",
}
F1_SUFFIX = "intermediate.dense.weight"
F2_SUFFIX = "output.dense.weight"

def add_project_to_path(project_root):
    if project_root and project_root not in sys.path:
        sys.path.append(project_root)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def suffix_for(param_type):
    if param_type in QKVO_SUFFIXES:
        return QKVO_SUFFIXES[param_type]
    if param_type == "f1":
        return F1_SUFFIX
    if param_type == "f2":
        return F2_SUFFIX
    raise KeyError(param_type)


def find_key(state_or_mask, layer_idx, suffix):
    expected = "bert.encoder.layer.%d.%s" % (layer_idx, suffix)
    if expected in state_or_mask:
        return expected
    matches = [
        key
        for key in state_or_mask
        if ("bert.encoder.layer.%d." % layer_idx) in key and key.endswith(suffix)
    ]
    if len(matches) != 1:
        raise KeyError(
            "Cannot uniquely find layer=%d suffix=%s, matches=%s"
            % (layer_idx, suffix, matches[:8])
        )
    return matches[0]


def get_target_parameters(model):
    params = {}
    for layer_idx in range(NUM_LAYERS):
        layer = model.bert.encoder.layer[layer_idx]
        params[(layer_idx, "q")] = layer.attention.self.query.weight
        params[(layer_idx, "k")] = layer.attention.self.key.weight
        params[(layer_idx, "v")] = layer.attention.self.value.weight
        params[(layer_idx, "o")] = layer.attention.output.dense.weight
        params[(layer_idx, "f1")] = layer.intermediate.dense.weight
        params[(layer_idx, "f2")] = layer.output.dense.weight
    return params


def load_model(args, device, model_dir):
    from transformer.configuration_bert import BertConfig
    from transformer.modeling_bert_quant import (
        BertForSequenceClassification as QuantBertForSequenceClassification,
    )

    config = BertConfig.from_pretrained(str(model_dir))
    config.num_labels = args.num_labels
    model = QuantBertForSequenceClassification.from_pretrained(
        str(model_dir), config=copy.deepcopy(config)
    )
    model.to(device)
    model.eval()
    return model


def inject_unprotected_sign_errors(args, device, model_dir, error_rate, rng_seed):
    import injector

    model = load_model(args, device, model_dir)
    seed_everything(rng_seed)
    bit_seed = 0
    print(
        "Injecting raw model sign errors: BER=%.4f rng_seed=%d bit_seed=0"
        % (error_rate, rng_seed)
    )
    with torch.no_grad():
        for matrix in get_target_parameters(model).values():
            if matrix.dtype != torch.float32:
                raise TypeError(
                    "injector float32 mode requires float32 weights, got %s"
                    % matrix.dtype
                )
            values = matrix.detach().cpu().numpy().reshape(-1).copy()
            size = int(values.size)
            injector.injector_bit(
                values,
                size,
                int(size * error_rate),
                bit_seed,
                "float32",
                mode="random",
            )
            restored = torch.from_numpy(values.reshape(tuple(matrix.shape)))
            matrix.copy_(restored.to(matrix.device))
    return model


def build_eval_dataloader(args, tokenizer, label_list, output_mode):
    from utils_glue import convert_examples_to_features, get_tensor_data, processors

    processor = processors[args.task_name.lower()]()
    examples = processor.get_dev_examples(args.data_dir)
    features = convert_examples_to_features(
        examples,
        label_list,
        args.max_seq_length,
        tokenizer,
        output_mode,
    )
    data, labels = get_tensor_data(output_mode, features)
    sampler = SequentialSampler(data)
    loader = DataLoader(data, sampler=sampler, batch_size=args.batch_size)
    return loader, labels, examples


def setup_eval_args(args):
    from transformer.tokenization import BertTokenizer
    from utils_glue import default_params, output_modes, processors

    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)
    if args.batch_size is None:
        args.batch_size = default_params[task_name]["batch_size"]
    if args.max_seq_length is None:
        args.max_seq_length = default_params[task_name]["max_seq_length"]
    processor = processors[task_name]()
    label_list = processor.get_labels()
    args.num_labels = len(label_list)
    args.output_mode = output_modes[task_name]
    tokenizer = BertTokenizer.from_pretrained(args.teacher_model, do_lower_case=True)
    return build_eval_dataloader(args, tokenizer, label_list, args.output_mode)


def unpack_batch(batch, device):
    batch = tuple(item.to(device) if torch.is_tensor(item) else item for item in batch)
    if len(batch) >= 4:
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
        }
        labels = batch[3]
    elif len(batch) == 3:
        inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
        labels = batch[2]
    elif len(batch) == 2:
        inputs = {"input_ids": batch[0]}
        labels = batch[1]
    else:
        raise ValueError("Unsupported batch length: %d" % len(batch))
    return inputs, labels


def get_logits(outputs):
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (tuple, list)):
        for item in outputs:
            if torch.is_tensor(item) and item.ndim >= 2:
                return item
        return outputs[0]
    return outputs


def evaluate_accuracy(model, dataloader, device, tag):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = unpack_batch(batch, device)
            try:
                outputs = model(**inputs)
            except TypeError:
                outputs = model(
                    inputs["input_ids"],
                    inputs.get("token_type_ids"),
                    inputs.get("attention_mask"),
                )
            predictions = torch.argmax(get_logits(outputs), dim=-1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    accuracy = correct / float(total) if total else 0.0
    print("%s: acc=%.6f correct=%d total=%d" % (tag, accuracy, correct, total))
    return accuracy


def resolve_artifacts(result_dir, explicit_model_dir, explicit_mask_path, mask_name):
    result_dir = Path(result_dir)
    if explicit_mask_path:
        mask_path = Path(explicit_mask_path)
    else:
        mask_candidates = (
            [result_dir / mask_name]
            if mask_name
            else [
                result_dir / "best_prefix_mask.pt",
                result_dir / "best_physical_prefix_mask.pt",
            ]
        )
        mask_path = next((path for path in mask_candidates if path.exists()), None)
        if mask_path is None:
            raise FileNotFoundError(
                "Cannot find prefix mask. Tried: %s"
                % [str(path) for path in mask_candidates]
            )
    if not mask_path.exists():
        raise FileNotFoundError("Missing prefix mask: %s" % mask_path)

    if explicit_model_dir:
        model_dir = Path(explicit_model_dir)
    else:
        model_candidates = [
            result_dir / "permuted_model",
            result_dir / "permuted_attention_ffn_model",
            result_dir,
        ]
        model_dir = next(
            (
                path
                for path in model_candidates
                if (path / "pytorch_model.bin").exists()
            ),
            None,
        )
        if model_dir is None:
            raise FileNotFoundError(
                "Cannot find permuted model. Tried: %s"
                % [str(path) for path in model_candidates]
            )
    if not (model_dir / "pytorch_model.bin").exists():
        raise FileNotFoundError("Missing pytorch_model.bin in %s" % model_dir)
    return model_dir, mask_path


def load_mask(mask_path):
    mask = torch.load(str(mask_path), map_location="cpu")
    if not isinstance(mask, dict):
        raise TypeError("Expected mask dict, got %s" % type(mask))
    return mask


def prefix_unit_count(selected, param_type):
    selected = selected.bool()
    if param_type in ROW_PREFIX_TYPES:
        full_units = selected.all(dim=1)
        reconstructed = full_units[:, None].expand_as(selected)
    else:
        full_units = selected.all(dim=0)
        reconstructed = full_units[None, :].expand_as(selected)
    if not torch.equal(selected, reconstructed):
        raise ValueError(
            "%s mask is not made of complete prefix %s"
            % (param_type, "rows" if param_type in ROW_PREFIX_TYPES else "columns")
        )
    count = int(full_units.sum().item())
    expected = torch.zeros_like(full_units)
    expected[:count] = True
    if not torch.equal(full_units, expected):
        raise ValueError("%s mask is not a contiguous prefix" % param_type)
    return count


def build_prefix_layouts(clean_model, mask):
    params = get_target_parameters(clean_model)
    layouts = []
    protected_bits = 0
    total_target_bits = 0
    stream_offset = 0

    for layer_idx in range(NUM_LAYERS):
        for param_type in TYPES:
            key = (layer_idx, param_type)
            parameter = params[key]
            mask_key = find_key(mask, layer_idx, suffix_for(param_type))
            selected = mask[mask_key].detach().cpu().bool()
            if tuple(selected.shape) != tuple(parameter.shape):
                raise ValueError(
                    "%s shape mismatch: mask=%s parameter=%s"
                    % (mask_key, tuple(selected.shape), tuple(parameter.shape))
                )
            count = prefix_unit_count(selected, param_type)
            if param_type in ROW_PREFIX_TYPES:
                orientation = "rows"
                unit_width = int(parameter.shape[1])
            else:
                orientation = "columns"
                unit_width = int(parameter.shape[0])
            bits = int(count * unit_width)
            layouts.append(
                {
                    "key": key,
                    "orientation": orientation,
                    "prefix_units": count,
                    "unit_width": unit_width,
                    "offset": stream_offset,
                    "bits": bits,
                }
            )
            stream_offset += bits
            protected_bits += int(selected.sum().item())
            total_target_bits += int(selected.numel())
    return layouts, protected_bits, total_target_bits


def extract_prefix_stream(clean_model, layouts):
    params = get_target_parameters(clean_model)
    parts = []
    for layout in layouts:
        parameter = params[layout["key"]]
        count = layout["prefix_units"]
        if layout["orientation"] == "rows":
            signs = (parameter.detach().cpu()[:count, :] >= 0).reshape(-1)
        else:
            signs = (
                parameter.detach().cpu()[:, :count] >= 0
            ).transpose(0, 1).reshape(-1)
        parts.append(signs.to(torch.uint8).numpy())
    if parts:
        return np.concatenate(parts).astype(np.uint8, copy=False)
    return np.empty(0, dtype=np.uint8)


def bsc(bits, error_rate, rng):
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.size == 0:
        return bits.copy(), 0
    flips = rng.random(bits.shape) < error_rate
    out = np.bitwise_xor(bits, flips.astype(np.uint8))
    return out.astype(np.uint8, copy=False), int(flips.sum())


def algebraic_cross_check_positions(hash_checks):
    """Construct metadata-free degree-4 checks using modular permutations."""
    multipliers = (17, 29, 47, 71)
    offsets = (0, 43, 91, 137)
    round_offsets = (0, 19, 53, 101)
    checks = []
    for round_offset in round_offsets:
        for h in range(TILE_BITS):
            positions = [
                int(
                    (multipliers[j] * h + offsets[j] + round_offset)
                    % TILE_BITS
                )
                for j in range(HASH_DEGREE)
            ]
            if len(set(positions)) != HASH_DEGREE:
                used = set()
                repaired = []
                for j, pos in enumerate(positions):
                    candidate = pos
                    step = 2 * j + 1
                    while candidate in used:
                        candidate = (candidate + step) % TILE_BITS
                    used.add(candidate)
                    repaired.append(candidate)
                positions = repaired
            checks.append(positions)
            if len(checks) >= hash_checks:
                return checks
    raise ValueError(
        "Algebraic mode supports at most %d hash checks with current variants"
        % len(checks)
    )


def tile_check_positions(hash_checks):
    checks = []
    segments = TILE_SIZE // SEGMENT_LENGTH
    for row in range(TILE_SIZE):
        for segment in range(segments):
            start = segment * SEGMENT_LENGTH
            checks.append(
                [
                    row * TILE_SIZE + start + offset
                    for offset in range(SEGMENT_LENGTH)
                ]
            )
    for column in range(TILE_SIZE):
        for segment in range(segments):
            start = segment * SEGMENT_LENGTH
            checks.append(
                [
                    (start + offset) * TILE_SIZE + column
                    for offset in range(SEGMENT_LENGTH)
                ]
            )
    checks.extend(algebraic_cross_check_positions(hash_checks))
    return np.asarray(checks, dtype=np.int64)


def hash_graph_stats(hash_checks):
    local_check_count = 2 * TILE_SIZE * (TILE_SIZE // SEGMENT_LENGTH)
    hash_only = tile_check_positions(hash_checks)[local_check_count:]
    coverage = np.zeros(TILE_BITS, dtype=np.int64)
    pair_counts = {}
    duplicate_checks = 0
    seen_checks = set()
    for check in hash_only:
        key = tuple(sorted(int(x) for x in check))
        if key in seen_checks:
            duplicate_checks += 1
        seen_checks.add(key)
        for pos in check:
            coverage[int(pos)] += 1
        for i in range(len(check)):
            for j in range(i + 1, len(check)):
                pair = tuple(sorted((int(check[i]), int(check[j]))))
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
    repeated_pairs = sum(1 for value in pair_counts.values() if value > 1)
    max_pair_repeat = max(pair_counts.values()) if pair_counts else 0
    return {
        "hash_coverage_min": int(coverage.min()) if coverage.size else 0,
        "hash_coverage_max": int(coverage.max()) if coverage.size else 0,
        "hash_coverage_mean": float(coverage.mean()) if coverage.size else 0.0,
        "hash_coverage_zero_bits": int((coverage == 0).sum()),
        "hash_duplicate_checks": int(duplicate_checks),
        "hash_repeated_pairs": int(repeated_pairs),
        "hash_max_pair_repeat": int(max_pair_repeat),
    }


def encode_tile_parity(clean_tile_flat, checks):
    clean_tile_flat = np.asarray(clean_tile_flat, dtype=np.uint8)
    parity = np.zeros(len(checks), dtype=np.uint8)
    for idx, positions in enumerate(checks):
        parity[idx] = int(np.bitwise_xor.reduce(clean_tile_flat[positions]))
    return parity


def decode_tile_noisy_bp(
    noisy_tile_flat,
    noisy_parity,
    check_positions,
    iterations,
    data_error_rate,
    parity_error_rate,
    damping,
    llr_clip,
):
    """Noisy-syndrome sum-product decoding for one tile.

    The observed data bits are BSC-corrupted with data_error_rate. The observed
    parity bits are also BSC-corrupted with parity_error_rate. For a check with
    observed parity z, the check-to-variable message is attenuated by
    (1-2*parity_error_rate), so noisy parity checks act as soft evidence rather
    than hard constraints.
    """
    noisy_tile_flat = np.asarray(noisy_tile_flat, dtype=np.uint8).reshape(-1)
    noisy_parity = np.asarray(noisy_parity, dtype=np.uint8).reshape(-1)
    check_positions = np.asarray(check_positions, dtype=np.int64)
    nbits = int(noisy_tile_flat.size)
    checks, degree = check_positions.shape
    if checks != int(noisy_parity.size):
        raise ValueError("Parity/check count mismatch")

    eps = 1e-6
    data_p = min(max(float(data_error_rate), eps), 1.0 - eps)
    parity_p = min(max(float(parity_error_rate), eps), 1.0 - eps)
    channel_llr_mag = math.log((1.0 - data_p) / data_p)
    channel_llr = np.where(
        noisy_tile_flat.astype(np.bool_),
        -channel_llr_mag,
        channel_llr_mag,
    ).astype(np.float32)

    syndrome_sign = np.where(
        noisy_parity.astype(np.bool_), -1.0, 1.0
    ).astype(np.float32)
    parity_reliability = np.float32(1.0 - 2.0 * parity_p)
    flat_positions = check_positions.reshape(-1)

    v_to_c = channel_llr[check_positions].astype(np.float32, copy=True)
    c_to_v = np.zeros_like(v_to_c, dtype=np.float32)
    damping = float(damping)
    llr_clip = float(llr_clip)

    for _ in range(iterations):
        tanh_messages = np.tanh(0.5 * v_to_c)
        new_c_to_v = np.empty_like(c_to_v)
        for edge_idx in range(degree):
            if degree == 1:
                prod_excluding = np.ones(checks, dtype=np.float32)
            else:
                others = [
                    tanh_messages[:, other_idx]
                    for other_idx in range(degree)
                    if other_idx != edge_idx
                ]
                prod_excluding = np.prod(np.stack(others, axis=1), axis=1)
            argument = parity_reliability * syndrome_sign * prod_excluding
            argument = np.clip(argument, -0.999999, 0.999999)
            new_c_to_v[:, edge_idx] = 2.0 * np.arctanh(argument)
        new_c_to_v = np.clip(new_c_to_v, -llr_clip, llr_clip)
        if damping > 0.0:
            c_to_v = (
                (1.0 - damping) * new_c_to_v + damping * c_to_v
            ).astype(np.float32)
        else:
            c_to_v = new_c_to_v.astype(np.float32)

        incoming = np.zeros(nbits, dtype=np.float32)
        np.add.at(incoming, flat_positions, c_to_v.reshape(-1))
        v_to_c = channel_llr[check_positions] + incoming[check_positions] - c_to_v
        v_to_c = np.clip(v_to_c, -llr_clip, llr_clip).astype(np.float32)

    posterior = channel_llr.copy()
    np.add.at(posterior, flat_positions, c_to_v.reshape(-1))
    decoded = (posterior < 0.0).astype(np.uint8)
    return decoded


def repair_prefix_stream_2d(source_bits, args):
    source_bits = np.asarray(source_bits, dtype=np.uint8).reshape(-1)
    tile_bits = TILE_BITS
    row_parity_per_tile = TILE_SIZE * (TILE_SIZE // SEGMENT_LENGTH)
    col_parity_per_tile = TILE_SIZE * (TILE_SIZE // SEGMENT_LENGTH)
    if row_parity_per_tile != 64 or col_parity_per_tile != 64:
        raise RuntimeError("Internal parity count mismatch.")

    source_count = int(source_bits.size)
    tiles = int(math.ceil(source_count / float(tile_bits))) if source_count else 0
    padded_count = tiles * tile_bits
    padded = np.zeros(padded_count, dtype=np.uint8)
    padded[:source_count] = source_bits
    decoded_padded = np.zeros_like(padded)

    rng = np.random.default_rng(args.fault_seed)
    total_data_channel_errors = 0
    total_parity_channel_errors = 0
    decoded_errors = 0
    tile_errors_before = 0
    tile_errors_after = 0

    parity_per_tile = row_parity_per_tile + col_parity_per_tile + args.hash_checks
    graph_stats = hash_graph_stats(args.hash_checks)
    print(
        "2D repair code: tile=%dx%d data=%d row=%d col=%d algebraic=%d "
        "total/tile=%d"
        % (
            TILE_SIZE,
            TILE_SIZE,
            tile_bits,
            row_parity_per_tile,
            col_parity_per_tile,
            args.hash_checks,
            tile_bits + parity_per_tile,
        )
    )
    print("Hash graph stats: %s" % json.dumps(graph_stats, sort_keys=True))
    progress_step = max(1, int(math.ceil(tiles / 10.0))) if tiles else 1
    next_progress = progress_step

    checks = tile_check_positions(args.hash_checks)
    for tile_index in range(tiles):
        start = tile_index * tile_bits
        stop = start + tile_bits
        clean_tile = padded[start:stop]
        clean_parity = encode_tile_parity(clean_tile, checks)
        noisy_tile, data_errors = bsc(clean_tile, args.error_rate, rng)
        noisy_parity, parity_errors = bsc(clean_parity, args.error_rate, rng)
        repaired = decode_tile_noisy_bp(
            noisy_tile,
            noisy_parity,
            checks,
            args.repair_iterations,
            args.error_rate,
            args.parity_error_rate,
            args.bp_damping,
            args.bp_llr_clip,
        )
        decoded_padded[start:stop] = repaired
        total_data_channel_errors += data_errors
        total_parity_channel_errors += parity_errors
        before_errors = int(np.bitwise_xor(noisy_tile, clean_tile).sum())
        after_errors = int(np.bitwise_xor(repaired, clean_tile).sum())
        decoded_errors += after_errors
        if before_errors:
            tile_errors_before += 1
        if after_errors:
            tile_errors_after += 1

        if tile_index + 1 >= next_progress or tile_index + 1 == tiles:
            print(
                "  2D repair progress: %d/%d tiles (%.1f%%)"
                % (tile_index + 1, tiles, 100.0 * (tile_index + 1) / max(1, tiles))
            )
            next_progress += progress_step

    decoded = decoded_padded[:source_count].copy()
    decoded_errors_real = int(np.bitwise_xor(decoded, source_bits).sum())
    information_padding_bits = int(padded_count - source_count)
    physical_prefix_bits = int(tiles * (tile_bits + parity_per_tile))
    parity_bits = int(tiles * parity_per_tile)
    stats = {
        "source_bits": source_count,
        "tiles": tiles,
        "tile_size": TILE_SIZE,
        "segment_length": SEGMENT_LENGTH,
        "tile_data_bits": tile_bits,
        "row_parity_bits": int(tiles * row_parity_per_tile),
        "column_parity_bits": int(tiles * col_parity_per_tile),
        "hash_parity_bits": int(tiles * args.hash_checks),
        "total_parity_bits": parity_bits,
        "information_padding_bits": information_padding_bits,
        "physical_prefix_bits_including_padding": physical_prefix_bits,
        "physical_prefix_bits_no_data_padding": int(source_count + parity_bits),
        "prefix_storage_multiplier_including_padding": (
            physical_prefix_bits / float(max(1, source_count))
        ),
        "prefix_storage_multiplier_no_data_padding": (
            (source_count + parity_bits) / float(max(1, source_count))
        ),
        "data_channel_errors_including_padding": int(total_data_channel_errors),
        "parity_channel_errors": int(total_parity_channel_errors),
        "actual_data_channel_ber_including_padding": (
            total_data_channel_errors / float(max(1, padded_count))
        ),
        "actual_parity_channel_ber": (
            total_parity_channel_errors / float(max(1, parity_bits))
        ),
        "decoded_errors_including_padding": int(decoded_errors),
        "decoded_errors": decoded_errors_real,
        "decoded_ber": decoded_errors_real / float(max(1, source_count)),
        "tile_error_rate_before": tile_errors_before / float(max(1, tiles)),
        "tile_error_rate_after": tile_errors_after / float(max(1, tiles)),
        "checks_per_tile": int(parity_per_tile),
        "cross_check_mode": "algebraic",
        "hash_degree": HASH_DEGREE,
        **graph_stats,
        "repair_iterations": int(args.repair_iterations),
        "decoder": "noisy_bp",
        "parity_error_rate_assumed_by_decoder": float(args.parity_error_rate),
        "bp_damping": float(args.bp_damping),
        "bp_llr_clip": float(args.bp_llr_clip),
    }
    print(json.dumps(stats, indent=2, sort_keys=True))
    return decoded, stats


def write_region_signs(candidate, clean, orientation, prefix_units, decoded_bits):
    if prefix_units <= 0:
        return
    decoded_bits = np.asarray(decoded_bits, dtype=np.uint8).reshape(-1)
    with torch.no_grad():
        if orientation == "rows":
            region_shape = tuple(candidate[:prefix_units, :].shape)
            signs = torch.from_numpy(decoded_bits.reshape(region_shape)).to(
                candidate.device
            ).bool()
            magnitudes = clean[:prefix_units, :].detach().abs()
            values = torch.where(signs, magnitudes, -magnitudes)
            candidate[:prefix_units, :].copy_(values)
        else:
            region_shape = (prefix_units, int(candidate.shape[0]))
            signs_t = torch.from_numpy(decoded_bits.reshape(region_shape)).to(
                candidate.device
            ).bool()
            signs = signs_t.transpose(0, 1)
            magnitudes = clean[:, :prefix_units].detach().abs()
            values = torch.where(signs, magnitudes, -magnitudes)
            candidate[:, :prefix_units].copy_(values)


def apply_decoded_prefix(candidate_model, clean_model, layouts, decoded_stream):
    candidate_params = get_target_parameters(candidate_model)
    clean_params = get_target_parameters(clean_model)
    for layout in layouts:
        start = layout["offset"]
        stop = start + layout["bits"]
        write_region_signs(
            candidate_params[layout["key"]],
            clean_params[layout["key"]],
            layout["orientation"],
            layout["prefix_units"],
            decoded_stream[start:stop],
        )


def evaluate_repaired_model(args, eval_dataloader, torch_device):
    model_dir, mask_path = resolve_artifacts(
        args.result_dir, args.permuted_model_dir, args.prefix_mask, args.mask_name
    )
    label = Path(args.result_dir).name
    print("\n" + "=" * 80)
    print("Result: %s" % label)
    print("Permuted model: %s" % model_dir)
    print("Prefix mask: %s" % mask_path)
    print("Sign-bit BER for data/parity/unprotected weights: %.4f" % args.error_rate)
    print("Fault seed: %d" % args.fault_seed)
    print("=" * 80)

    clean_model = load_model(args, torch_device, model_dir)
    mask = load_mask(mask_path)
    layouts, protected_bits, total_target_bits = build_prefix_layouts(
        clean_model, mask
    )
    source_bits = extract_prefix_stream(clean_model, layouts)
    if int(source_bits.size) != protected_bits:
        raise RuntimeError("Extracted prefix stream length does not match mask size")

    decoded_stream, repair_stats = repair_prefix_stream_2d(source_bits, args)
    repaired_model = inject_unprotected_sign_errors(
        args, torch_device, model_dir, args.error_rate, args.fault_seed
    )
    apply_decoded_prefix(repaired_model, clean_model, layouts, decoded_stream)
    repaired_acc = evaluate_accuracy(
        repaired_model,
        eval_dataloader,
        torch_device,
        "%s repaired model" % label,
    )

    physical_prefix_bits = repair_stats["physical_prefix_bits_including_padding"]
    physical_prefix_bits_no_pad = repair_stats["physical_prefix_bits_no_data_padding"]
    total_physical_target_bits = (
        total_target_bits - protected_bits + physical_prefix_bits
    )
    total_physical_target_bits_no_pad = (
        total_target_bits - protected_bits + physical_prefix_bits_no_pad
    )
    summary = {
        "result_dir": str(args.result_dir),
        "permuted_model_dir": str(model_dir),
        "prefix_mask": str(mask_path),
        "scheme": "prefix-only 16x16 row/column/algebraic parity repair",
        "protected_source_bits": int(protected_bits),
        "target_source_bits": int(total_target_bits),
        "mask_protection_rate": protected_bits / float(max(1, total_target_bits)),
        "error_rate": float(args.error_rate),
        "fault_seed": int(args.fault_seed),
        "cross_check_mode": "algebraic",
        "decoder": "noisy_bp",
        "repaired_acc": repaired_acc,
        "repair_stats": repair_stats,
        "extra_physical_bits_over_uncoded_target_including_padding": int(
            total_physical_target_bits - total_target_bits
        ),
        "extra_physical_bits_over_uncoded_target_no_data_padding": int(
            total_physical_target_bits_no_pad - total_target_bits
        ),
        "total_physical_target_bits_including_padding": int(total_physical_target_bits),
        "total_physical_target_bits_no_data_padding": int(
            total_physical_target_bits_no_pad
        ),
        "whole_target_storage_multiplier_including_padding": (
            total_physical_target_bits / float(max(1, total_target_bits))
        ),
        "whole_target_storage_multiplier_no_data_padding": (
            total_physical_target_bits_no_pad / float(max(1, total_target_bits))
        ),
    }

    print("\nStorage-cost summary")
    print("  protected source bits: %d" % protected_bits)
    print("  row parity bits: %d" % repair_stats["row_parity_bits"])
    print("  column parity bits: %d" % repair_stats["column_parity_bits"])
    print("  hash parity bits: %d" % repair_stats["hash_parity_bits"])
    print("  total parity bits: %d" % repair_stats["total_parity_bits"])
    print(
        "  encoded prefix physical bits incl. tile padding: %d"
        % repair_stats["physical_prefix_bits_including_padding"]
    )
    print(
        "  extra physical bits over uncoded target incl. padding: %d"
        % summary["extra_physical_bits_over_uncoded_target_including_padding"]
    )
    print(
        "  prefix multiplier incl. padding: %.6fx"
        % repair_stats["prefix_storage_multiplier_including_padding"]
    )
    print(
        "  whole target-weight storage multiplier incl. padding: %.6fx"
        % summary["whole_target_storage_multiplier_including_padding"]
    )
    print(
        "  decoded prefix BER after 2D repair: %.10f"
        % repair_stats["decoded_ber"]
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.result_dir) / "prefix_2d_repair_eval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / (
        "%s_prefix_2d_repair_algebraic_bp_hash%d_seed%d.json"
        % (label, args.hash_checks, args.fault_seed)
    )
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print("Saved result summary: %s" % result_path)

    del clean_model, repaired_model, mask
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate algebraic 2D parity protection with noisy belief "
            "propagation for physical prefixes in W1 BERT models."
        )
    )
    parser.add_argument("--project_root", default=".", type=str)
    parser.add_argument("--teacher_model", required=True, type=str)
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--task_name", default="SST-2", type=str)
    parser.add_argument("--max_seq_length", default=None, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--error_rate", default=0.1, type=float)
    parser.add_argument("--fault_seed", default=42, type=int)
    parser.add_argument("--result_dir", required=True, type=str)
    parser.add_argument("--permuted_model_dir", default=None, type=str)
    parser.add_argument("--prefix_mask", default=None, type=str)
    parser.add_argument("--mask_name", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)

    parser.add_argument("--hash_checks", default=256, type=int)
    parser.add_argument("--repair_iterations", default=20, type=int)
    parser.add_argument(
        "--parity_error_rate",
        default=None,
        type=float,
        help="Parity-bit BER assumed by noisy BP. Defaults to --error_rate.",
    )
    parser.add_argument(
        "--bp_damping",
        default=0.25,
        type=float,
        help="Damping for noisy BP check-to-variable messages.",
    )
    parser.add_argument(
        "--bp_llr_clip",
        default=12.0,
        type=float,
        help="Absolute LLR clipping value for noisy BP.",
    )
    args = parser.parse_args()

    if not 0.0 < args.error_rate < 0.5:
        raise ValueError("error_rate must be in (0, 0.5)")
    if args.parity_error_rate is None:
        args.parity_error_rate = args.error_rate
    if not 0.0 < args.parity_error_rate < 0.5:
        raise ValueError("parity_error_rate must be in (0, 0.5)")
    max_hash_checks = 4 * TILE_BITS
    if not 1 <= args.hash_checks <= max_hash_checks:
        raise ValueError("hash_checks must be in [1, %d]" % max_hash_checks)
    if args.repair_iterations <= 0:
        raise ValueError("repair_iterations must be positive")
    if not 0.0 <= args.bp_damping < 1.0:
        raise ValueError("bp_damping must be in [0, 1)")
    if args.bp_llr_clip <= 0.0:
        raise ValueError("bp_llr_clip must be positive")
    return args


def main():
    args = parse_args()
    add_project_to_path(args.project_root)
    seed_everything(args.fault_seed)
    torch_device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print("PyTorch evaluation device: %s" % torch_device)
    print("All data/parity/unprotected sign bits use BER=%.4f" % args.error_rate)
    print("Fault seed: %d" % args.fault_seed)
    print("Cross-check mode: algebraic")
    print("Decoder: noisy_bp")
    print(
        "Noisy BP assumes parity-bit BER=%.4f, damping=%.3f, llr_clip=%.2f"
        % (args.parity_error_rate, args.bp_damping, args.bp_llr_clip)
    )
    print("Project convention: bit_seed=0 injects sign-bit faults.")
    eval_dataloader, _, _ = setup_eval_args(args)
    evaluate_repaired_model(args, eval_dataloader, torch_device)


if __name__ == "__main__":
    main()
