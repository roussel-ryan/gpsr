from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from gpsr.beams import ResNNTransform
from gpsr.train import train_gpsr, train_gpsr_multistep


class DummyGPSR(torch.nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        self.beam_generator = SimpleNamespace(transformer=transformer)
        self.readout = torch.nn.Linear(2, 2)


class TestTrainGPSR:
    @patch("gpsr.train.L.Trainer")
    @patch("gpsr.train.ModelCheckpoint")
    def test_train_gpsr_honors_supplied_dirpath(self, mock_checkpoint, mock_trainer):
        mock_logger = Mock()
        mock_logger.log_dir = "/tmp/logger"
        gpsr_model = Mock()
        dataloader = Mock()

        train_gpsr(
            gpsr_model,
            dataloader,
            logger=mock_logger,
            dirpath="/tmp/custom-checkpoints",
        )

        mock_checkpoint.assert_called_once()
        assert mock_checkpoint.call_args.kwargs["dirpath"] == "/tmp/custom-checkpoints"


class TestTrainGPSRMultistep:
    def test_requires_skip_connection(self):
        model = DummyGPSR(ResNNTransform(n_hidden=2, width=10, phase_space_dim=2))

        with pytest.raises(
            ValueError,
            match="train_gpsr_multistep requires a ResNNTransform with use_skip_connection=True",
        ):
            train_gpsr_multistep(model, Mock())

    @patch("gpsr.train.train_gpsr")
    def test_stage_two_trains_full_model(self, mock_train_gpsr):
        transformer = ResNNTransform(
            n_hidden=2,
            width=10,
            phase_space_dim=2,
            use_skip_connection=True,
        )
        model = DummyGPSR(transformer)
        recorded_requires_grad = []

        def capture_call(gpsr_model, *args, **kwargs):
            recorded_requires_grad.append(
                {
                    "linear": [p.requires_grad for p in transformer.linear_parameters],
                    "non_linear": [
                        p.requires_grad
                        for p in gpsr_model.parameters()
                        if id(p) not in {id(param) for param in transformer.linear_parameters}
                    ],
                }
            )
            return SimpleNamespace(gpsr_model=gpsr_model)

        mock_train_gpsr.side_effect = capture_call

        train_gpsr_multistep(model, Mock())

        assert len(recorded_requires_grad) == 2
        assert all(recorded_requires_grad[0]["linear"])
        assert not any(recorded_requires_grad[0]["non_linear"])
        assert all(recorded_requires_grad[1]["linear"])
        assert all(recorded_requires_grad[1]["non_linear"])
