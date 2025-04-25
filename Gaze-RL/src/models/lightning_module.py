# import pytorch_lightning as pl
# import torch.nn as nn
# from torchmetrics import MeanSquaredError
# from torch.optim import Adam

# from src.models.gaze_predictor import GazePredictor


# class GazeLightningModule(pl.LightningModule):
#     def __init__(self, config):
#         super().__init__()
#         self.save_hyperparameters()
#         self.config = config
#         self.model = GazePredictor()
#         self.mse_metric = MeanSquaredError()
#         self.loss_fn = nn.MSELoss()

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         imgs, heatmaps = batch
#         preds = self(imgs)
#         loss = self.loss_fn(preds, heatmaps)
        
#         self.log("train_loss", loss, prog_bar=True)
#         self.log("train_mse", self.mse_metric(preds, heatmaps))
#         return loss

#     def validation_step(self, batch, batch_idx):
#         imgs, heatmaps = batch
#         preds = self(imgs)
#         self.log("val_mse", self.mse_metric(preds, heatmaps))

#     def configure_optimizers(self):
#         return Adam(self.parameters(), 
#                   lr=float(self.config["training"]["lr"]),
#                   weight_decay=float(self.config["training"]["weight_decay"]))

#     def predict_step(self, batch, batch_idx):
#         imgs, _ = batch
#         return self(imgs)


import pytorch_lightning as pl
import torch.nn as nn
from torchmetrics import MeanSquaredError
from torch.optim import Adam

from src.models.gaze_predictor import GazePredictor


class GazeLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = GazePredictor()
        self.mse_metric = MeanSquaredError()
        self.loss_fn = nn.MSELoss()

        # To accumulate loss and metrics for averaging
        self.train_loss_epoch = 0.0
        self.val_mse_epoch = 0.0
        self.train_step_count = 0
        self.val_step_count = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, heatmaps = batch
        preds = self(imgs)
        loss = self.loss_fn(preds, heatmaps)
        
        # Log running training metrics
        self.train_loss_epoch += loss.item()
        self.train_step_count += 1
        return loss

    def on_train_epoch_end(self):
        # Log average training loss at the end of the epoch
        avg_train_loss = self.train_loss_epoch / self.train_step_count
        self.log("epoch_train_loss", avg_train_loss, prog_bar=True)

        # Reset counters for the next epoch
        self.train_loss_epoch = 0.0
        self.train_step_count = 0

    def validation_step(self, batch, batch_idx):
        imgs, heatmaps = batch
        preds = self(imgs)
        
        # Accumulate validation metrics
        self.val_mse_epoch += self.mse_metric(preds, heatmaps)
        self.val_step_count += 1

    def on_validation_epoch_end(self):
        # Log average validation metrics at the end of the epoch
        avg_val_mse = self.val_mse_epoch / self.val_step_count
        self.log("epoch_val_mse", avg_val_mse, prog_bar=True)

        # Reset counters for the next epoch
        self.val_mse_epoch = 0.0
        self.val_step_count = 0

    def configure_optimizers(self):
        return Adam(self.parameters(), 
                  lr=float(self.config["training"]["lr"]),
                  weight_decay=float(self.config["training"]["weight_decay"]))

    def predict_step(self, batch, batch_idx):
        imgs, _ = batch
        return self(imgs)
