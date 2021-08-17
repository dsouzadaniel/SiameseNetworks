# Libraries
import os
import loaders, models
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import BCELoss

train_omg = loaders.OmniglotDataset(dataset_info='./train_dataset_temp.txt')
train_omg_dl = DataLoader(train_omg, batch_size=32)

val_omg = loaders.OmniglotDataset(dataset_info='./val_dataset_temp.txt')
val_omg_dl = DataLoader(val_omg, batch_size=32)

# for b in omg_dl:
#     (img1s, img2s), targets = b
#     breaks
#
# print(img1s.shape)
# print(img2s.shape)
# print(targets.shape)
#
#
# loss_fn = BCELoss()
# # Model
# siam = models.SiameseNetwork()
# # print(siam)
#
# preds  = siam(img1=img1s, img2=img2s)
#
# print(preds.shape)
#
# loss = loss_fn(input=preds, target=targets)
# print(loss)

siam = models.SiameseNetworkLightning()
# trainer = pl.Trainer(overfit_batches=1, log_every_n_steps=100)
# trainer.fit(siam, train_omg_dl)

if os.path.exists('pretrained_models'):
    os.makedirs('pretrained_models')
# Calllbacks

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="./pretrained_models",
    filename="siam-{epoch:02d}-{val_loss:.2f}",
    save_top_k=1,
    mode="min",
)



trainer = pl.Trainer(log_every_n_steps=len(train_omg_dl), callbacks=[checkpoint_callback])
trainer.fit(siam, train_omg_dl, val_omg_dl)
