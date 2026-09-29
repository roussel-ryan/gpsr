"""Tests for ``gpsr.lume.model`` helpers."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lume_cheetah")
from tensordict import TensorDict

from gpsr.lume.model import _merge_settings


class TestMergeSettings:
    def test_settings_and_constants_are_combined(self):
        merged = _merge_settings(
            {"Q1:BCTRL": torch.tensor([1.0, 2.0])},
            {"SOL:BCTRL": torch.tensor(0.5)},
        )

        assert sorted(merged) == ["Q1:BCTRL", "SOL:BCTRL"]

    @pytest.mark.parametrize("constants", [None, {}])
    def test_no_constants_leaves_the_settings_alone(self, constants):
        settings = {"Q1:BCTRL": torch.tensor([1.0, 2.0])}

        assert _merge_settings(settings, constants) == settings

    def test_a_tensordict_of_settings_is_accepted(self):
        # The training batch yields settings as a TensorDict, not a plain dict.
        settings = TensorDict({"Q1:BCTRL": torch.tensor([1.0, 2.0])}, batch_size=[2])

        assert sorted(_merge_settings(settings, {"SOL:BCTRL": torch.tensor(0.5)})) == [
            "Q1:BCTRL",
            "SOL:BCTRL",
        ]

    def test_a_shared_key_is_rejected(self):
        # Without this the constant would win, leaving the scan looking flat.
        with pytest.raises(ValueError, match=r"share the key\(s\) \['Q1:BCTRL'\]"):
            _merge_settings(
                {"Q1:BCTRL": torch.tensor([1.0, 2.0])},
                {"Q1:BCTRL": torch.tensor(5.0)},
            )

    def test_a_shared_key_is_rejected_even_when_it_looks_like_a_scan(self):
        # The dangerous case: the merged tensor keeps the right shape, so nothing
        # downstream can tell the scan was replaced by a repeated constant.
        with pytest.raises(ValueError, match="Q1:BCTRL"):
            _merge_settings(
                {"Q1:BCTRL": torch.tensor([1.0, 2.0, 3.0])},
                {"Q1:BCTRL": torch.tensor([5.0, 5.0, 5.0])},
            )


class TestMergeSettingsDevice:
    """The constants are moved; the settings are left where the caller put them.

    ``meta`` stands in for a second device so these run without a GPU.
    """

    @pytest.mark.parametrize(
        "constant", [torch.tensor(5.0), torch.tensor([5.0]), 5.0], ids=type
    )
    def test_constants_are_moved(self, constant):
        merged = _merge_settings(
            {"Q1:BCTRL": torch.tensor([1.0, 2.0])},
            {"SOL:BCTRL": constant},
            device=torch.device("meta"),
        )

        assert merged["SOL:BCTRL"].device.type == "meta"

    def test_settings_are_left_alone(self):
        settings = {"Q1:BCTRL": torch.tensor([1.0, 2.0])}

        merged = _merge_settings(settings, None, device=torch.device("meta"))

        assert merged["Q1:BCTRL"].device.type == "cpu"

    def test_no_device_moves_nothing(self):
        # A lattice holding no tensors has no device to move to.
        constant = torch.tensor(5.0)

        merged = _merge_settings({}, {"SOL:BCTRL": constant}, device=None)

        assert merged["SOL:BCTRL"] is constant
