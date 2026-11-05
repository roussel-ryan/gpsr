from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from gpsr.beams import ResNNTransform
from gpsr.train import ResNNTransformFinetuning, train_gpsr


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


class TestResNNTransformFinetuning:
    @pytest.fixture
    def pl_module(self):
        transformer = ResNNTransform(
            n_hidden=2,
            width=10,
            phase_space_dim=2,
            use_skip_connection=True,
        )
        with torch.no_grad():
            transformer.alpha.fill_(0.5)
        return SimpleNamespace(gpsr_lume_model=DummyGPSR(transformer))

    @pytest.mark.parametrize("hook", ["freeze_before_training", "finetune_function"])
    def test_requires_resnn_transform(self, hook):
        pl_module = SimpleNamespace(gpsr_lume_model=DummyGPSR(torch.nn.Linear(2, 2)))
        callback = ResNNTransformFinetuning()

        with pytest.raises(
            ValueError,
            match="Expected the transformer to be an instance of ResNNTransform",
        ):
            if hook == "freeze_before_training":
                callback.freeze_before_training(pl_module)
            else:
                callback.finetune_function(pl_module, 0, Mock())

    def test_freezes_and_zeros_only_residual_branch(self, pl_module):
        model = pl_module.gpsr_lume_model
        transformer = model.beam_generator.transformer
        original_state = {
            name: value.clone() for name, value in model.state_dict().items()
        }

        ResNNTransformFinetuning().freeze_before_training(pl_module)

        assert not any(p.requires_grad for p in transformer.resnet.parameters())
        assert not transformer.alpha.requires_grad
        assert transformer.alpha.item() == 0.0
        assert all(p.requires_grad for p in transformer.linear_parameters)
        assert all(p.requires_grad for p in model.readout.parameters())
        for name, value in model.state_dict().items():
            if name != "beam_generator.transformer.alpha":
                torch.testing.assert_close(value, original_state[name])

    def test_can_disable_initial_freezing(self, pl_module):
        model = pl_module.gpsr_lume_model
        next(model.beam_generator.transformer.resnet.parameters()).requires_grad_(False)
        original_requires_grad = [p.requires_grad for p in model.parameters()]
        original_state = {
            name: value.clone() for name, value in model.state_dict().items()
        }

        ResNNTransformFinetuning(freeze_at_start=False).freeze_before_training(
            pl_module
        )

        assert [p.requires_grad for p in model.parameters()] == original_requires_grad
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original_state[name])

    @pytest.mark.parametrize("unfreeze_epoch", [0, 2])
    def test_unfreezes_and_adds_parameters_at_scheduled_epoch(
        self, pl_module, unfreeze_epoch
    ):
        model = pl_module.gpsr_lume_model
        transformer = model.beam_generator.transformer
        callback = ResNNTransformFinetuning(unfreeze_epoch=unfreeze_epoch)
        callback.freeze_before_training(pl_module)
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        initial_ids = {id(p) for p in optimizer.param_groups[0]["params"]}
        assert initial_ids == {
            id(p) for p in [*transformer.linear_parameters, *model.readout.parameters()]
        }

        for epoch in range(unfreeze_epoch):
            callback.finetune_function(pl_module, epoch, optimizer)
            assert not any(p.requires_grad for p in transformer.resnet_parameters)
            assert transformer.alpha.item() == 0.0
            assert len(optimizer.param_groups) == 1
            assert {id(p) for p in optimizer.param_groups[0]["params"]} == initial_ids

        callback.finetune_function(pl_module, unfreeze_epoch, optimizer)

        assert all(p.requires_grad for p in model.parameters())
        assert transformer.alpha.item() == 0.0
        assert len(optimizer.param_groups) == 3
        assert {id(p) for p in optimizer.param_groups[0]["params"]} == initial_ids
        assert {id(p) for p in optimizer.param_groups[1]["params"]} == {
            id(p) for p in transformer.resnet.parameters()
        }
        assert [id(p) for p in optimizer.param_groups[2]["params"]] == [
            id(transformer.alpha)
        ]

        callback.finetune_function(pl_module, unfreeze_epoch + 1, optimizer)

        assert len(optimizer.param_groups) == 3
        optimizer_ids = [
            id(p) for group in optimizer.param_groups for p in group["params"]
        ]
        assert len(optimizer_ids) == len(set(optimizer_ids))
        assert set(optimizer_ids) == {id(p) for p in model.parameters()}
