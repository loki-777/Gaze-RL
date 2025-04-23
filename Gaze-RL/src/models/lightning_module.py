import pytorch_lightning as pl
import torch.nn as nn
from torchmetrics import MeanSquaredError
from torch.optim import Adam

from gaze_predictor import GazePredictor


class GazeLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = GazePredictor()
        self.mse_metric = MeanSquaredError()
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, heatmaps = batch
        preds = self(imgs)
        loss = self.loss_fn(preds, heatmaps)
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mse", self.mse_metric(preds, heatmaps))
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, heatmaps = batch
        preds = self(imgs)
        self.log("val_mse", self.mse_metric(preds, heatmaps))

    def configure_optimizers(self):
        return Adam(self.parameters(), 
                  lr=self.config["lr"],
                  weight_decay=self.config["weight_decay"])

    def predict_step(self, batch, batch_idx):
        imgs, _ = batch
        return self(imgs)