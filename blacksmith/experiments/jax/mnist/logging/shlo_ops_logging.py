# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import re

import jax
import jax.numpy as jnp
from jax import export


class ExportSHLO:
    @staticmethod
    def export_and_get_ops(exported, print_text, print_stablehlo):
        if print_stablehlo:
            print(print_text)
            print(exported.mlir_module())

        pattern = r"stablehlo\.(\w+)"
        operations = re.findall(pattern, exported.mlir_module())
        unique_operations = sorted(set(operations))
        print(f"{print_text} Unique Operations:", ", ".join(unique_operations))

    @staticmethod
    def export_fwd_train_to_StableHLO_and_get_ops(forward_pass, state, input_shape, print_stablehlo=True):
        exported_forward_tr: export.Exported = export.export(forward_pass)(state.params, jnp.ones(input_shape))
        ExportSHLO.export_and_get_ops(exported_forward_tr, "Forward Pass StableHLO (Train):", print_stablehlo)

    @staticmethod
    def export_fwd_tst_to_StableHLO_and_get_ops(eval_step, state, input_shape, print_stablehlo=True):
        exported_forward_tst: export.Exported = export.export(eval_step)(state.params, jnp.ones(input_shape))
        ExportSHLO.export_and_get_ops(exported_forward_tst, "Forward Pass StableHLO (Test):", print_stablehlo)

    @staticmethod
    def export_bwd_to_StableHLO_and_get_ops(backward_pass, state, input_shape, print_stablehlo=True):
        exported_backward: export.Exported = export.export(backward_pass)(
            state.params, jnp.ones(input_shape), jnp.ones((1,), dtype=jnp.int32)
        )
        ExportSHLO.export_and_get_ops(exported_backward, "Backward Pass StableHLO:", print_stablehlo)

    @staticmethod
    def export_loss_to_StableHLO_and_get_ops(loss_fn, y, print_stablehlo=True):
        exported_loss: export.Exported = export.export(loss_fn)(y, jnp.ones((1,), dtype=jnp.int32))
        ExportSHLO.export_and_get_ops(exported_loss, "Loss Function StableHLO:", print_stablehlo)

    @staticmethod
    def export_optimizer_to_StableHLO_and_get_ops(update_params, state, grads, print_stablehlo=True):
        exported_optimizer: export.Exported = export.export(update_params)(state, grads)
        ExportSHLO.export_and_get_ops(exported_optimizer, "Optimizer StableHLO:", print_stablehlo)
