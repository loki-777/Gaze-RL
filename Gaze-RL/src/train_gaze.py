import torch
import yaml
import argparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from utils import SALICONDataset
from src.models.lightning_module import GazeLightningModule

def get_dataloaders(config):
    train_dataset = SALICONDataset(
        img_dir=config["data"]["img_dir"],
        heatmap_dir=config["data"]["heatmap_dir"]
    )
    return DataLoader(train_dataset, 
                     batch_size=config["data"]["batch_size"],
                     shuffle=True,
                     num_workers=4)

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Data
    train_loader = get_dataloaders(config)

    # Model
    model = GazeLightningModule(config)

    # Training
    logger = WandbLogger(project="gaze-prediction", name=config["logging"]["experiment_name"])
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if config["training"]["use_gpu"] else "auto",
        max_epochs=config["training"]["epochs"],
        logger=logger,
        callbacks=[
            ModelCheckpoint(dirpath=config["logging"]["checkpoint_dir"]),
            LearningRateMonitor()
        ]
    )
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gaze_config.yaml")
    args = parser.parse_args()
    main(args.config)