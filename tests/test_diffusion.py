"""Tests for ``src.diffusion.mean_flat``.

``src/diffusion.py`` imports ``k_diffusion`` at module level and subclasses out
of it, so it cannot be sensibly stubbed -- skip if it is not installed.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("k_diffusion")

from src.diffusion import mean_flat  # noqa: E402


class TestMeanFlat:
    @pytest.mark.parametrize("shape", [(3, 5), (3, 5, 4), (3, 5, 4, 2)])
    def test_reduces_everything_except_the_batch_axis(self, shape):
        x = torch.randn(shape)

        out = mean_flat(x)

        assert out.shape == (shape[0],)
        torch.testing.assert_close(out, x.reshape(shape[0], -1).mean(dim=1))

    def test_matches_a_hand_computed_value(self):
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 8.0]]])
        torch.testing.assert_close(mean_flat(x), torch.tensor([2.5, 2.0]))

    def test_batch_of_one(self):
        x = torch.ones(1, 6, 4)
        torch.testing.assert_close(mean_flat(x), torch.tensor([1.0]))

    def test_each_row_is_reduced_independently(self):
        x = torch.stack([torch.full((4, 3), float(i)) for i in range(5)])
        torch.testing.assert_close(mean_flat(x), torch.arange(5, dtype=torch.float32))

    def test_does_not_modify_its_input(self):
        x = torch.randn(3, 5, 4)
        before = x.clone()
        mean_flat(x)
        torch.testing.assert_close(x, before)

    def test_preserves_dtype(self):
        x = torch.randn(3, 5, 4, dtype=torch.float64)
        assert mean_flat(x).dtype == torch.float64

    def test_gradient_is_spread_evenly(self):
        x = torch.zeros(2, 3, 4, requires_grad=True)

        mean_flat(x).sum().backward()

        expected = torch.full((2, 3, 4), 1.0 / 12)
        torch.testing.assert_close(x.grad, expected)

    def test_works_on_a_non_contiguous_tensor(self):
        x = torch.randn(3, 5, 4).transpose(1, 2)
        assert not x.is_contiguous()
        torch.testing.assert_close(mean_flat(x), x.reshape(3, -1).mean(dim=1))

    # NB: mean_flat on a 1-D tensor becomes ``tensor.mean(dim=[])``, whose
    # meaning changed across torch versions (reduce-all vs reduce-nothing).
    # The loss code only ever passes >=2-D tensors, so that case is left
    # untested rather than pinned to one torch release.
