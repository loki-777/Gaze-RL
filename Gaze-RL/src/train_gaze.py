import torch
import yaml
import argparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.utils import *
from src.models.lightning_module import GazeLightningModule

import os, glob

torch.backends.nnpack.enabled = False

def get_dataloaders(config):
    train_dataset = SALICONDataset(
        img_dir=config["data"]["img_dir"],
        heatmap_dir=config["data"]["heatmap_dir"]
    )

    val_dataset = SALICONDataset(
        img_dir=config["data"]["img_dir"].replace("train", "val"),
        heatmap_dir=config["data"]["heatmap_dir"].replace("train", "val")
    )
    return (DataLoader(train_dataset, 
                     batch_size=config["data"]["batch_size"],
                     shuffle=True,
                     num_workers=4),
            DataLoader(val_dataset, 
                     batch_size=config["data"]["batch_size"],
                     shuffle=False,
                     num_workers=4)
    )

def main(config_path, run_test=False):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Data
    train_loader, val_loader = get_dataloaders(config)

    if run_test:
        ckpt = glob.glob(os.path.join(config["logging"]["checkpoint_dir"], "UNET", "*.ckpt"))[0]
        model = GazeLightningModule.load_from_checkpoint(ckpt)
        for batch in val_loader:
            model.eval()
            # model = model.to(torch.float32)
            # predictions = model.predict_step(batch, batch_idx=0)
            images, gts = batch
            images = images.to(torch.float32)
            predictions = model(images)
            visualize_predictions(images[:5, :, :, :], gts[:5, :, :, :], predictions[:5, :, :, :], num_samples=5)
            # print(images.shape, gts.shape, predictions.shape)
            break
        return
    
    # Model
    model = GazeLightningModule(config)

    # Training
    logger = WandbLogger(project="gaze-prediction", name=config["logging"]["experiment_name"])
    trainer = pl.Trainer(
        # accelerator="gpu" if torch.cuda.is_available() else "cpu",
        accelerator="cpu",
        # devices=torch.cuda.device_count() if config["training"]["use_gpu"] else "auto",
        max_epochs=config["training"]["epochs"],
        logger=logger,
        callbacks=[
            ModelCheckpoint(dirpath=config["logging"]["checkpoint_dir"]),
            LearningRateMonitor()
        ]
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gaze_config.yaml")
    parser.add_argument("--test", action='store_true')
    args = parser.parse_args()
    main(args.config, args.test)