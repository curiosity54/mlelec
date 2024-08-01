import lightning.pytorch as pl

class LoggingCallback(pl.Callback):
    def __init__(self, log_every_n_epochs=1):
        super().__init__()
        self.train_loss = None
        self.val_loss = None
        self.rmse_eigenvalues = None
        self.rmse_ard = None
        self.log_every_n_epochs = log_every_n_epochs

    def on_fit_start(self, trainer, pl_module):
        # Ensure log_every_n_steps and check_val_every_n_epoch are taken from the Trainer config
        self.log_every_n_steps = trainer.log_every_n_steps
        self.check_val_every_n_epoch = trainer.check_val_every_n_epoch

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Store train loss every log_every_n_steps
        if (trainer.global_step + 1) % self.log_every_n_steps == 0:
            train_loss = outputs['loss'] if isinstance(outputs, dict) else outputs
            self.train_loss = train_loss.item()

    def on_validation_epoch_end(self, trainer, pl_module):
        # Store validation loss at the end of each validation epoch
        val_loss = trainer.callback_metrics.get('val_loss')
        rmse_eigenvalues = trainer.callback_metrics.get('rmse_eigenvalues')
        rmse_ard = trainer.callback_metrics.get('rmse_atom_resolved_density')
        if val_loss:
            self.val_loss = val_loss.item()
            self.rmse_eigenvalues = rmse_eigenvalues
            self.rmse_ard = rmse_ard

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        log_train = epoch % self.log_every_n_epochs == 0
        log_val = epoch % self.check_val_every_n_epoch == 0

        # Print train loss at the first epoch
        if epoch == 1:
            train_loss = trainer.callback_metrics.get('train_loss')
            if train_loss:
                self.train_loss = train_loss.item()
            print(f"Epoch {epoch}, Train Loss: {self.train_loss}")

        # If both train and val should be logged in the same epoch, ensure it is printed only once
        elif log_val:
            # Log validation, and train loss if available
            val_loss = self.val_loss
            train_loss = trainer.callback_metrics.get('train_loss')
            if val_loss:
                if train_loss:
                    self.train_loss = train_loss.item()
                print(f"Epoch {epoch}, Train Loss: {self.train_loss}, Validation Loss: {val_loss}, "
                      f"RMSE Eigenvalues: {self.rmse_eigenvalues}, RMSE ARD: {self.rmse_ard}")
                self.train_loss = None  # Reset train_loss after printing
        elif log_train:
            # Log train loss
            train_loss = trainer.callback_metrics.get('train_loss')
            if train_loss:
                self.train_loss = train_loss.item()
            print(f"Epoch {epoch}, Train Loss: {self.train_loss}")
            self.train_loss = None  # Reset train_loss after printing
