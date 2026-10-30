"""Regression tests for array-like adjacency matrix conversion."""

import numpy as np
import pytest
import torch

from gnm.utils.convert_datatypes import np_to_tensor


@pytest.mark.parametrize(
    "input_matrix",
    [
        np.array([[0, 1], [1, 0]], dtype=np.int64),
        np.array([[0, 1], [1, 0]], dtype=np.float64),
        [[0, 1], [1, 0]],
        ((0, 1), (1, 0)),
        torch.tensor([[0, 1], [1, 0]], dtype=torch.float64),
    ],
)
def test_unbatched_array_like_inputs(input_matrix):
    result = np_to_tensor(input_matrix, device="cpu")

    assert result.shape == (1, 2, 2)
    assert result.dtype == torch.float32
    assert result.device.type == "cpu"
    torch.testing.assert_close(result[0], torch.tensor([[0.0, 1.0], [1.0, 0.0]]))


def test_batched_input_keeps_its_batch_axis():
    matrices = np.arange(18, dtype=np.float64).reshape(2, 3, 3)

    result = np_to_tensor(matrices, device=torch.device("cpu"))

    assert result.shape == (2, 3, 3)
    torch.testing.assert_close(result, torch.tensor(matrices, dtype=torch.float32))


def test_numpy_array_with_negative_strides():
    matrix = np.array([[0.0, 1.0], [2.0, 3.0]])[::-1]

    result = np_to_tensor(matrix, device="cpu")

    torch.testing.assert_close(result[0], torch.tensor([[2.0, 3.0], [0.0, 1.0]]))


def test_tensor_input_retains_gradient_flow():
    matrix = torch.ones((2, 2), requires_grad=True)

    result = np_to_tensor(matrix, device="cpu")
    result.sum().backward()

    torch.testing.assert_close(matrix.grad, torch.ones((2, 2)))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_reviewer_numpy_conversion_on_mps():
    connectome_array = np.ones((2, 3, 3), dtype=np.float64)
    distance_matrix = np.eye(3, dtype=np.int64)
    device = torch.device("mps")

    connectome_tensor = np_to_tensor(connectome_array, device=device)
    distance_tensor = np_to_tensor(distance_matrix, device=device)

    assert connectome_tensor.shape == (2, 3, 3)
    assert distance_tensor.shape == (1, 3, 3)
    assert connectome_tensor.device.type == distance_tensor.device.type == "mps"
    assert connectome_tensor.dtype == distance_tensor.dtype == torch.float32
    torch.testing.assert_close(distance_tensor.cpu()[0], torch.eye(3))


@pytest.mark.parametrize("invalid", [1.0, [1, 2, 3], np.zeros((1, 2, 3, 4))])
def test_invalid_number_of_dimensions(invalid):
    with pytest.raises(ValueError, match="2D or 3D"):
        np_to_tensor(invalid, device="cpu")
