# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import lorax


def split_trainable_frozen(lora_params, lora_spec):
    """
    Split LoRA parameters into trainable and frozen pytrees.
    Trainable: Only LoRA a,b matrices from LoraWeight objects
    Frozen: Everything else (original weights w, alpha, and regular frozen params)
    """
    trainable_params = {}
    frozen_params = {}

    def split_param(param, spec_value, path_parts):
        if isinstance(param, lorax.LoraWeight):
            # For LoraWeight: only a,b are trainable
            trainable_params[".".join(path_parts)] = {"a": param.a, "b": param.b}
            frozen_params[".".join(path_parts)] = {"w": param.w, "alpha": param.alpha}
        else:
            # Regular parameters go to frozen (they should all be spec=0)
            frozen_params[".".join(path_parts)] = param

    def traverse_tree(params_tree, spec_tree, path=[]):
        if isinstance(params_tree, dict):
            for key in params_tree:
                traverse_tree(params_tree[key], spec_tree[key], path + [key])
        else:
            split_param(params_tree, spec_tree, path)

    traverse_tree(lora_params, lora_spec)

    print(f"📊 Split completed:")
    print(f"   Trainable params: {len(trainable_params)} LoRA matrix pairs")
    print(f"   Frozen params: {len(frozen_params)} weight groups")

    return trainable_params, frozen_params


def merge_trainable_frozen(trainable_params, frozen_params):
    """
    Merge trainable and frozen pytrees back into full LoRA parameter tree.
    """
    merged_params = {}

    # First add all frozen regular parameters
    for path, param in frozen_params.items():
        if isinstance(param, dict) and "w" in param and "alpha" in param:
            # This is a frozen LoraWeight component - will be merged with trainable
            continue
        else:
            # Regular frozen parameter
            keys = path.split(".")
            current = merged_params
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = param

    # Now merge LoraWeight objects
    for path, trainable in trainable_params.items():
        frozen_lora = frozen_params[path]  # Should have 'w' and 'alpha'

        # Reconstruct LoraWeight
        lora_weight = lorax.LoraWeight(
            w=frozen_lora["w"], a=trainable["a"], b=trainable["b"], alpha=frozen_lora["alpha"]
        )

        # Place in merged tree
        keys = path.split(".")
        current = merged_params
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = lora_weight

    return merged_params
