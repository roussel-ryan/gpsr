import torch
from torch import nn
from torchensemble import SnapshotEnsembleRegressor
from torchensemble.utils import set_module, io


class CustomSnapshotRegressor(SnapshotEnsembleRegressor):
    def create_estimator(self):
        self.estimator = self._make_estimator()

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def fit(
        self,
        train_loader,
        lr_clip=None,
        epochs=100,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
    ):
        self._validate_parameters(lr_clip, epochs, log_interval)
        self.n_outputs = self._decide_n_outputs(train_loader)

        # Set the optimizer and scheduler
        optimizer = self.optimizer

        scheduler = self._set_scheduler(optimizer, epochs * len(train_loader))

        # Check the training criterion
        if not hasattr(self, "_criterion"):
            self._criterion = nn.MSELoss()

        # Utils
        best_loss = float("inf")
        counter = 0  # a counter on generating snapshots
        total_iters = 0
        n_iters_per_estimator = epochs * len(train_loader) // self.n_estimators

        # Training loop
        self.estimator.train()
        for epoch in range(epochs):
            for batch_idx, elem in enumerate(train_loader):

                data, target = io.split_data_target(elem, self.device)

                # Clip the learning rate
                optimizer = self._clip_lr(optimizer, lr_clip)

                optimizer.zero_grad()
                output = self.estimator(*data)
                loss = self._criterion(output, target)
                loss.backward()
                optimizer.step()

                # Print training status
                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        msg = (
                            "lr: {:.5f} | Epoch: {:03d} | Batch: {:03d}"
                            " | Loss: {:.5f}"
                        )
                        self.logger.info(
                            msg.format(
                                optimizer.param_groups[0]["lr"],
                                epoch,
                                batch_idx,
                                loss,
                            )
                        )
                        if self.tb_logger:
                            self.tb_logger.add_scalar(
                                "snapshot_ensemble/Train_Loss",
                                loss,
                                total_iters,
                            )

                # Snapshot ensemble updates the learning rate per iteration
                # instead of per epoch.
                scheduler.step()
                counter += 1
                total_iters += 1

            if counter % n_iters_per_estimator == 0:
                # Generate and save the snapshot
                snapshot = self._make_estimator()
                snapshot.load_state_dict(self.estimator.state_dict())
                self.estimators_.append(snapshot)

                msg = "Save the snapshot model with index: {}"
                self.logger.info(msg.format(len(self.estimators_) - 1))

            # Validation after each snapshot model being generated
            if test_loader and counter % n_iters_per_estimator == 0:
                self.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for _, elem in enumerate(test_loader):
                        data, target = io.split_data_target(elem, self.device)
                        output = self.forward(*data)
                        val_loss += self._criterion(output, target)
                    val_loss /= len(test_loader)

                    if val_loss < best_loss:
                        best_loss = val_loss
                        if save_model:
                            io.save(self, save_dir, self.logger)

                    msg = (
                        "n_estimators: {} | Validation Loss: {:.5f} |"
                        " Historical Best: {:.5f}"
                    )
                    self.logger.info(
                        msg.format(len(self.estimators_), val_loss, best_loss)
                    )
                    if self.tb_logger:
                        self.tb_logger.add_scalar(
                            "snapshot_ensemble/Validation_Loss",
                            val_loss,
                            len(self.estimators_),
                        )

        # prep for saving model (can't save criterion that depends on model object)
        self._criterion = None

        if save_model and not test_loader:
            io.save(self, save_dir, self.logger)