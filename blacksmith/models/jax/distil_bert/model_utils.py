# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from flax.core.frozen_dict import freeze, unfreeze

# Recursively walks through parameter tree, extracting 'embeddings' into frozen dict.
def _recursive_extract_embeddings(node):
    if not isinstance(node, dict):
        return node, None
    trainable = {}
    frozen = {}
    for k, v in node.items():
        if k == "embeddings":
            frozen[k] = v
            continue
        t_sub, f_sub = _recursive_extract_embeddings(v)
        trainable[k] = t_sub
        if f_sub is not None:
            frozen[k] = f_sub
    if len(frozen) == 0:
        frozen = None
    return trainable, frozen


# Split model params into trainable (everything except embeddings) and frozen (embeddings only).
def split_params(params):
    p = unfreeze(params)
    trainable_root = {}
    frozen_root = {}
    for topk, topv in p.items():
        t_sub, f_sub = _recursive_extract_embeddings(topv)
        trainable_root[topk] = t_sub
        if f_sub is not None:
            frozen_root[topk] = f_sub
    return freeze(trainable_root), freeze(frozen_root)


# Merge trainable and frozen params back into complete parameter tree for model inference.
def combine_params(trainable, frozen):
    t = unfreeze(trainable)
    f = unfreeze(frozen)

    def _merge(t_node, f_node):
        if f_node is None:
            return t_node
        for k, v in f_node.items():
            if k in t_node:
                if isinstance(v, dict) and isinstance(t_node[k], dict):
                    t_node[k] = _merge(t_node[k], v)
                else:
                    t_node[k] = v
            else:
                t_node[k] = v
        return t_node

    merged = _merge(t, f)
    return freeze(merged)
