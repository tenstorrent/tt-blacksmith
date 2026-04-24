# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import logging
import re

import jax
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec

logger = logging.getLogger(__name__)

PartitionRule = tuple[str, PartitionSpec]


def _path_to_str(path: tuple) -> str:
    """Join a JAX tree key-path into a dot-separated string."""
    parts: list[str] = []
    for key in path:
        if hasattr(key, "key"):
            parts.append(str(key.key))
        elif hasattr(key, "idx"):
            parts.append(str(key.idx))
        else:
            parts.append(str(key))
    return ".".join(parts)


def build_param_partition_specs(
    state,
    partition_rules: list[PartitionRule],
    *,
    default: PartitionSpec = PartitionSpec(),
) -> dict:
    """Assign a PartitionSpec to every leaf of state, matching the first rule that applies.

    Args:
        state: An nnx.State or arbitrary pytree whose leaves are arrays.
        partition_rules: Ordered (regex, PartitionSpec) list; first match wins.
        default: Fallback spec for leaves that match no rule.

    Returns:
        A pytree mirroring state with PartitionSpec leaves.
    """
    flat, treedef = jax.tree_util.tree_flatten_with_path(state)
    specs: list[PartitionSpec] = []
    for path, _ in flat:
        name = _path_to_str(path)
        matched = default
        for pattern, spec in partition_rules:
            if re.search(pattern, name):
                matched = spec
                break
        specs.append(matched)
    return jax.tree_util.tree_unflatten(treedef, specs)


def build_named_shardings(
    specs_tree,
    mesh: Mesh,
):
    """Map a pytree of PartitionSpec into NamedSharding for the given mesh."""
    return jax.tree.map(lambda s: NamedSharding(mesh, s), specs_tree)


def shard_pytree(pytree, shardings_tree):
    """Place every leaf according to the matching sharding."""
    return jax.tree.map(
        lambda x, s: jax.device_put(x, s),
        pytree,
        shardings_tree,
    )


def easydel_partition_specs_for_lora(
    model: nnx.Module,
    mesh: Mesh,
    *,
    lora_default: PartitionSpec = PartitionSpec(),
    frozen_default: PartitionSpec = PartitionSpec(),
):
    """Split an EasyDel model into LoRA and frozen state and build their shardings.

    Returns:
        (graphdef, lora_state, frozen_state, (lora_shardings, frozen_shardings))
    """
    graphdef, lora_state, frozen_state = nnx.split(model, nnx.LoRAParam, ...)

    has_rules = (
        hasattr(model, "config") and hasattr(model.config, "partition_rules") and callable(model.config.partition_rules)
    )

    if has_rules:
        rules = model.config.partition_rules()
        lora_specs = build_param_partition_specs(lora_state, rules, default=lora_default)
        frozen_specs = build_param_partition_specs(frozen_state, rules, default=frozen_default)
    else:
        lora_specs = jax.tree.map(lambda _: lora_default, lora_state)
        frozen_specs = jax.tree.map(lambda _: frozen_default, frozen_state)

    lora_shardings = build_named_shardings(lora_specs, mesh)
    frozen_shardings = build_named_shardings(frozen_specs, mesh)

    return (
        graphdef,
        lora_state,
        frozen_state,
        (lora_shardings, frozen_shardings),
    )
