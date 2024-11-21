import re
import jax
from jax import export
import jax.numpy as jnp

class ExportSHLO:
    @staticmethod
    def export_fwd_train_to_StableHLO_and_get_ops(forward_pass, state, input_shape, print_stablehlo=True):
        exported_forward_tr: export.Exported = export.export(forward_pass)(state.params, jnp.ones(input_shape))
        if(print_stablehlo):
            print("Forward Pass StableHLO (Train):")
            print(exported_forward_tr.mlir_module())

        import re
        pattern = r"stablehlo\.(\w+)"
        operations_forward = re.findall(pattern, exported_forward_tr.mlir_module())
        unique_operations_forward = sorted(set(operations_forward))
        print("Unique Operations in Forward Pass (Training):", ", ".join(unique_operations_forward))
    
    @staticmethod
    def export_fwd_tst_to_StableHLO_and_get_ops(eval_step, state, input_shape, print_stablehlo=True):
        exported_forward_tst: export.Exported = export.export(eval_step)(state.params, jnp.ones(input_shape))
        if(print_stablehlo):
            print("Forward Pass StableHLO (Test):")
            print(exported_forward_tst.mlir_module())

        import re
        pattern = r"stablehlo\.(\w+)"
        operations_forward_tst = re.findall(pattern, exported_forward_tst.mlir_module())
        unique_operations_forward_tst = sorted(set(operations_forward_tst))
        print("Unique Operations in Forward Pass (Test):", ", ".join(unique_operations_forward_tst))
    
    @staticmethod
    def export_bwd_to_StableHLO_and_get_ops(backward_pass, state, input_shape, print_stablehlo=True):
        exported_backward: export.Exported = export.export(backward_pass)(state.params, jnp.ones(input_shape), jnp.ones((1,), dtype=jnp.int32))        
        if(print_stablehlo):
            print("Backward Pass StableHLO:")
            print(exported_backward.mlir_module())

        import re
        pattern = r"stablehlo\.(\w+)"
        operations_backward = re.findall(pattern, exported_backward.mlir_module())
        unique_operations_backward = sorted(set(operations_backward))
        print("Unique Operations in Backward Pass:", ", ".join(unique_operations_backward))\
    
    @staticmethod
    def export_loss_to_StableHLO_and_get_ops(func_optax_loss, output_shape, print_stablehlo=True):
        exported_loss: export.Exported = export.export(func_optax_loss)(output_shape, jnp.ones((1,), dtype=jnp.int32))
        if(print_stablehlo):
            print("Loss Function StableHLO:")
            print(exported_loss.mlir_module())

        import re
        pattern = r"stablehlo\.(\w+)"
        operations_loss = re.findall(pattern, exported_loss.mlir_module())
        unique_operations_loss = sorted(set(operations_loss))
        print("Unique Operations in Loss Function:", ", ".join(unique_operations_loss))

    @staticmethod
    def export_optimizer_to_StableHLO_and_get_ops(update_params, state, grads, print_stablehlo=True):
        exported_optimizer: export.Exported = export.export(update_params)(state, grads)
        if(print_stablehlo):
            print("Optimizer StableHLO:")
            print(exported_optimizer.mlir_module())

        import re
        pattern = r"stablehlo\.(\w+)"
        operations_optimizer = re.findall(pattern, exported_optimizer.mlir_module())
        unique_operations_optimizer = sorted(set(operations_optimizer))
        print("Unique Operations in Optimizer:", ", ".join(unique_operations_optimizer))



