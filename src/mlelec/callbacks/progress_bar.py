from lightning.pytorch.callbacks import TQDMProgressBar
from tqdm.auto import tqdm

class ProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self.train_loss = None
        self.val_loss = None
        self.rmse_eigenvalues = None
        self.rmse_ard = None
        self.main_progress_bar = None

    def on_train_start(self, trainer, pl_module):
        self.main_progress_bar = tqdm(total=trainer.max_epochs, desc="Training")

    def on_train_epoch_end(self, trainer, pl_module):
        self.main_progress_bar.update(1)
        self.main_progress_bar.set_postfix(epoch=trainer.current_epoch, train_loss=self.train_loss, val_loss=self.val_loss, rmse_eigenvalues=self.rmse_eigenvalues, rmse_ard=self.rmse_ard)

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val_loss')
        rmse_eigenvalues = trainer.callback_metrics.get('rmse_eigenvalues')
        rmse_ard = trainer.callback_metrics.get('rmse_atom_resolved_density')
        if val_loss:
            self.val_loss = val_loss.item()
            self.rmse_eigenvalues = rmse_eigenvalues.item() if rmse_eigenvalues else None
            self.rmse_ard = rmse_ard.item() if rmse_ard else None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        train_loss = outputs['loss'] if isinstance(outputs, dict) else outputs
        self.train_loss = train_loss.item()
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            self.main_progress_bar.set_postfix(epoch=trainer.current_epoch, train_loss=self.train_loss, val_loss=self.val_loss, rmse_eigenvalues=self.rmse_eigenvalues, rmse_ard=self.rmse_ard)

    def on_train_end(self, trainer, pl_module):
        self.main_progress_bar.close()
