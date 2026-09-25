# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch.nn.functional as F

try:
    from torch_geometric.data import Data

    from blacksmith.experiments.torch.BOUNTIES.graphsage_reddit.batching import (
        masked_correct,
        masked_cross_entropy,
        prepare_neighbor_batch,
        sampled_graph_capacity,
    )
    from blacksmith.models.torch.graphsage.graphsage import GraphSAGE
except ModuleNotFoundError as error:
    if error.name and error.name.startswith("torch_geometric"):
        pytest.skip("GraphSAGE tests require torch_geometric", allow_module_level=True)
    raise

pytestmark = [
    pytest.mark.push,
    pytest.mark.n300,
    pytest.mark.torch,
    pytest.mark.single_chip,
    pytest.mark.pyg,
]


def test_sampled_graph_capacity_is_conservative() -> None:
    assert sampled_graph_capacity(2, [2, 1]) == (13, 10)


def test_prepare_neighbor_batch_uses_isolated_sentinel_padding() -> None:
    batch = Data(
        x=torch.tensor([[1.0], [2.0], [3.0]]),
        edge_index=torch.tensor([[1], [0]]),
        y=torch.tensor([2, 1, 0]),
    )
    batch.batch_size = 1

    prepared = prepare_neighbor_batch(
        batch=batch,
        device=torch.device("cpu"),
        seed_capacity=2,
        num_neighbors=[1],
        static_shapes=True,
    )

    assert prepared.x.shape == (5, 1)
    assert prepared.edge_index.shape == (2, 2)
    assert torch.equal(prepared.edge_index[:, -1], torch.tensor([4, 4]))
    assert torch.equal(prepared.target_y, torch.tensor([2, 0]))
    assert torch.equal(prepared.target_mask, torch.tensor([True, False]))
    assert prepared.target_capacity == 2
    assert prepared.target_count == 1


def test_masked_metrics_ignore_padded_targets() -> None:
    logits = torch.tensor([[0.0, 0.0, 3.0], [8.0, 0.0, 0.0]])
    targets = torch.tensor([2, 1])
    mask = torch.tensor([True, False])

    expected_loss = F.cross_entropy(logits[:1], targets[:1])
    assert torch.allclose(masked_cross_entropy(logits, targets, mask), expected_loss)
    assert masked_correct(logits, targets, mask).item() == 1.0


@pytest.mark.parametrize("use_spmm", [False, True])
def test_static_padding_does_not_change_seed_outputs(use_spmm: bool) -> None:
    torch.manual_seed(19)
    batch = Data(
        x=torch.randn(3, 4),
        edge_index=torch.tensor([[1], [0]]),
        y=torch.tensor([1, 0, 1]),
    )
    batch.batch_size = 1
    model = GraphSAGE(4, 5, 2, dropout=0.0, use_spmm=use_spmm).eval()

    expected = model(batch.x, batch.edge_index)[:1]
    prepared = prepare_neighbor_batch(
        batch=batch,
        device=torch.device("cpu"),
        seed_capacity=2,
        num_neighbors=[1],
        static_shapes=True,
    )
    actual = model(prepared.x, prepared.edge_index)[:1]

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("use_spmm", [False, True])
def test_static_padding_preserves_masked_loss_and_gradients(use_spmm: bool) -> None:
    torch.manual_seed(23)
    reference_model = GraphSAGE(4, 5, 2, dropout=0.0, use_spmm=use_spmm)
    padded_model = GraphSAGE(4, 5, 2, dropout=0.0, use_spmm=use_spmm)
    padded_model.load_state_dict(reference_model.state_dict())

    reference_x = torch.randn(3, 4, requires_grad=True)
    padded_x = reference_x.detach().clone().requires_grad_(True)
    edge_index = torch.tensor([[1], [0]])
    targets = torch.tensor([1, 0, 1])

    reference_loss = F.cross_entropy(reference_model(reference_x, edge_index)[:1], targets[:1])
    batch = Data(x=padded_x, edge_index=edge_index, y=targets)
    batch.batch_size = 1
    prepared = prepare_neighbor_batch(
        batch=batch,
        device=torch.device("cpu"),
        seed_capacity=2,
        num_neighbors=[1],
        static_shapes=True,
    )
    padded_logits = padded_model(prepared.x, prepared.edge_index)[: prepared.target_capacity]
    padded_loss = masked_cross_entropy(padded_logits, prepared.target_y, prepared.target_mask)

    torch.testing.assert_close(padded_loss, reference_loss)
    reference_loss.backward()
    padded_loss.backward()
    torch.testing.assert_close(padded_x.grad, reference_x.grad)

    reference_parameters = dict(reference_model.named_parameters())
    padded_parameters = dict(padded_model.named_parameters())
    for name in reference_parameters:
        torch.testing.assert_close(padded_parameters[name].grad, reference_parameters[name].grad)
