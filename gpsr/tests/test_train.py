from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from gpsr.beams import ResNNTransform
from gpsr.train import train_gpsr, train_gpsr_multistep


class DummyBeamGenerator(torch.nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer


class DummyGPSR(torch.nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.beam_generator = DummyBeamGenerator(transformer)
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

    @patch("gpsr.train.L.Trainer")
    @patch("gpsr.train.ModelCheckpoint")
    def test_train_gpsr_preserves_empty_dirpath(self, mock_checkpoint, mock_trainer):
        mock_logger = Mock()
        mock_logger.log_dir = "/tmp/logger"

        train_gpsr(Mock(), Mock(), logger=mock_logger, dirpath="")

        mock_checkpoint.assert_called_once()
        assert mock_checkpoint.call_args.kwargs["dirpath"] == ""


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
        with torch.no_grad():
            transformer.alpha.fill_(0.5)
        model = DummyGPSR(transformer)
        recorded_requires_grad = []
        recorded_alpha_values = []
        stage_one_trainable_ids = {
            id(param) for param in transformer.linear_parameters
        } | {id(transformer.log_output_scale)}

        def capture_call(gpsr_model, *args, **kwargs):
            recorded_alpha_values.append(transformer.alpha.item())
            recorded_requires_grad.append(
                {
                    "linear": [p.requires_grad for p in transformer.linear_parameters],
                    "output_scale": transformer.log_output_scale.requires_grad,
                    "alpha": transformer.alpha.requires_grad,
                    "non_linear": [
                        p.requires_grad
                        for p in gpsr_model.parameters()
                        if id(p) not in stage_one_trainable_ids
                    ],
                }
            )
            return SimpleNamespace(gpsr_model=gpsr_model)

        mock_train_gpsr.side_effect = capture_call

        train_gpsr_multistep(model, Mock())

        assert len(recorded_requires_grad) == 2
        assert recorded_alpha_values == [0.0, 0.5]
        assert all(recorded_requires_grad[0]["linear"])
        assert recorded_requires_grad[0]["output_scale"]
        assert not recorded_requires_grad[0]["alpha"]
        assert not any(recorded_requires_grad[0]["non_linear"])
        assert all(recorded_requires_grad[1]["linear"])
        assert recorded_requires_grad[1]["output_scale"]
        assert recorded_requires_grad[1]["alpha"]
        assert all(recorded_requires_grad[1]["non_linear"])

    @patch("gpsr.train.train_gpsr", side_effect=RuntimeError("stage failure"))
    def test_stage_one_failure_restores_state(self, mock_train_gpsr):
        transformer = ResNNTransform(
            n_hidden=2,
            width=10,
            phase_space_dim=2,
            use_skip_connection=True,
        )
        with torch.no_grad():
            transformer.alpha.fill_(0.5)
        model = DummyGPSR(transformer)
        original_requires_grad = [p.requires_grad for p in model.parameters()]

        with pytest.raises(RuntimeError, match="stage failure"):
            train_gpsr_multistep(model, Mock())

        assert [p.requires_grad for p in model.parameters()] == original_requires_grad
        assert transformer.alpha.item() == pytest.approx(0.5)
