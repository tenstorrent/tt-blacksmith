# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

try:
    from torch_geometric.nn import SAGEConv

    from blacksmith.models.torch.graphsage.graphsage import GraphSAGE
    from blacksmith.models.torch.graphsage.spmm_graphsage import SpMMGraphSAGEConv
except ModuleNotFoundError as error:
    if error.name and error.name.startswith("torch_geometric"):
        pytest.skip("GraphSAGE tests require torch_geometric", allow_module_level=True)
    raise


def _edge_index_with_duplicates_and_isolated_targets() -> torch.Tensor:
    # 0 -> 1 and 2 -> 3 are duplicated. Nodes 0, 2, and 5 have no incoming
    # neighbors, so their aggregate must be zero before the linear transforms.
    return torch.tensor(
        [
            [0, 0, 1, 2, 2, 2, 4],
            [1, 1, 1, 3, 3, 4, 4],
        ],
        dtype=torch.long,
    )


def _assert_parameter_gradients_match(reference, actual) -> None:
    reference_parameters = dict(reference.named_parameters())
    actual_parameters = dict(actual.named_parameters())
    assert reference_parameters.keys() == actual_parameters.keys()

    for name in reference_parameters:
        torch.testing.assert_close(
            actual_parameters[name].grad,
            reference_parameters[name].grad,
            rtol=1e-10,
            atol=1e-12,
        )


def test_spmm_conv_matches_pyg_forward_and_gradients() -> None:
    torch.manual_seed(7)
    edge_index = _edge_index_with_duplicates_and_isolated_targets()

    reference = SAGEConv(4, 3).double()
    actual = SpMMGraphSAGEConv(4, 3).double()
    actual.load_state_dict(reference.state_dict())

    reference_input = torch.randn(6, 4, dtype=torch.double, requires_grad=True)
    actual_input = reference_input.detach().clone().requires_grad_(True)
    loss_weights = torch.randn(6, 3, dtype=torch.double)

    reference_output = reference(reference_input, edge_index)
    actual_output = actual(actual_input, edge_index)
    torch.testing.assert_close(actual_output, reference_output, rtol=1e-10, atol=1e-12)

    # An isolated target receives lin_l(0) + lin_r(x), including lin_l's bias.
    isolated_expected = actual.lin_l(actual_input.new_zeros(1, 4))[0] + actual.lin_r(actual_input[5])
    torch.testing.assert_close(actual_output[5], isolated_expected, rtol=1e-10, atol=1e-12)

    (reference_output * loss_weights).sum().backward()
    (actual_output * loss_weights).sum().backward()

    torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=1e-10, atol=1e-12)
    _assert_parameter_gradients_match(reference, actual)


def test_graphsage_spmm_path_matches_stock_path() -> None:
    torch.manual_seed(11)
    edge_index = _edge_index_with_duplicates_and_isolated_targets()

    reference = GraphSAGE(4, 5, 3, dropout=0.0).double()
    actual = GraphSAGE(4, 5, 3, dropout=0.0, use_spmm=True).double()
    actual.load_state_dict(reference.state_dict())

    assert type(reference.conv1) is SAGEConv
    assert type(reference.conv2) is SAGEConv
    assert type(actual.conv1) is SpMMGraphSAGEConv
    assert type(actual.conv2) is SpMMGraphSAGEConv

    reference_input = torch.randn(6, 4, dtype=torch.double, requires_grad=True)
    actual_input = reference_input.detach().clone().requires_grad_(True)
    loss_weights = torch.randn(6, 3, dtype=torch.double)

    reference_output = reference(reference_input, edge_index)
    actual_output = actual(actual_input, edge_index)
    torch.testing.assert_close(actual_output, reference_output, rtol=1e-10, atol=1e-12)

    (reference_output * loss_weights).sum().backward()
    (actual_output * loss_weights).sum().backward()

    torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=1e-10, atol=1e-12)
    _assert_parameter_gradients_match(reference, actual)


def test_spmm_forward_and_backward_execute_without_scatter_ops() -> None:
    inputs = torch.randn(8, 4, requires_grad=True)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]],
        dtype=torch.long,
    )
    layer = SpMMGraphSAGEConv(4, 3)

    with torch.autograd.profiler.profile() as profile:
        layer(inputs, edge_index).sum().backward()

    operation_names = {event.name.lower() for event in profile.function_events}
    assert not any("scatter" in name for name in operation_names)
    assert any("matmul" in name or "mm" in name for name in operation_names)
