from models.model import Rater
from data.dataloader import get_dataloader
from utils.util import TorchHelper
import wandb
from tqdm import tqdm
import warnings
from sklearn.metrics import f1_score, recall_score, precision_score
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch import optim
from random import shuffle
import pandas as pd
import numpy as np
import json
import time
from torch import nn
import random
import torch
from torch.utils.data import DataLoader

import os
import pickle
import sys
import argparse


sys.path.append('../')
sys.path.append("../data")

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# fix random seed as 7
torch.manual_seed(7)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(7)
torch.backends.cudnn.enabled = False

random.seed(7)


# for metrics


warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='MPAA rating with BERT.')
parser.add_argument('--batch_size', type=int,
                    help='(int) batch size', default=16)

parser.add_argument('--max_epochs', type=int,
                    help='(int) max number of epochs', default=60)

parser.add_argument("--logger_type", type=str,
                    help="wandb or tensorboard", default="tensorboard")

parser.add_argument(
    "--lr", type=float, help="initial learning rate", default=1e-6)

parser.add_argument(
    "--base_dir", type=str, help="base directiory", default="/home/chaehyeong/MARS_hj/BERT_rating"
)


args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = args.batch_size
logger_type = args.logger_type


loss_weights1 = torch.Tensor([0.5669, 0.2913, 0.1418])


# run_mode = 'run'
run_mode = 'test'
# run_mode = 'test_resume'
criterian = nn.CrossEntropyLoss()
torch_helper = TorchHelper()


start_epoch = 0


max_epochs = args.max_epochs
learning_rate = args.lr
clip_grad = 0.5
weight_decay_val = 0
optimizer_type = 'adam'  # sgd

base_dir = args.base_dir
sys.path.append(os.path.join(base_dir, "data"))
base_log_dir = os.path.join(base_dir, "results/log")
log_dir = os.path.join(
    base_log_dir, f"lr_{learning_rate}_batch_size{batch_size}_")
summary = SummaryWriter(log_dir=os.path.join(base_dir, "results/log"))


partition_dict = json.load(
    open(os.path.join(base_dir, "data/partition.json"), 'r'))

if logger_type == "wandb":
    wandb.init()
    wandb.init(project="MPAA_rating_with_BERT")

    wandb.config.batch_size = batch_size
    wandb.config.learning_rate = learning_rate

    wandb.define_metric("train_step")
    wandb.define_metric("epoch")
    wandb.define_metric(
        "train_step/*", step_metric="train_step", summary="min")
    wandb.define_metric("train_epoch/*", step_metric="epoch")
    wandb.define_metric("train_score_macro/*", step_metric="epoch")
    wandb.define_metric("train_score_micro/*", step_metric="epoch")
    wandb.define_metric("train_score_weighted/*", step_metric="epoch")
    wandb.define_metric("val_epoch/*", step_metric="epoch", summary="max")
    wandb.define_metric("val_score_macro/*", step_metric="epoch")
    wandb.define_metric("val_score_micro/*", step_metric="epoch")
    wandb.define_metric("val_score_weighted/*", step_metric="epoch")

train_id_list = partition_dict['train']
val_id_list = partition_dict['val']
test_id_list = partition_dict['test']


# Learning rate scheduler
lr_schedule_active = False
reduce_on_plateau_lr_schdlr = torch.optim.lr_scheduler.ReduceLROnPlateau

# Creates the directory where the results, logs, and models will be dumped.
run_name = f'rating_with_BERT_lr{learning_rate}/'
description = ''

output_dir_path = os.path.join(base_dir, 'results', run_name)
if not os.path.exists(output_dir_path):
    os.mkdir(output_dir_path)


total_train_step_count = 0


# ------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------
def create_model(device):
    """
    Creates and returns the EmotionFlowModel.
    Moves to GPU if found any.
    :return:

    """
    model = Rater()

    model = nn.DataParallel(model)
    model.to(device)
    if run_mode == 'test':
        torch_helper.load_saved_model(model, output_dir_path + 'best.pth')
        print('model loaded')

    return model


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------
def train(model, optimizer, dataloader, device):
    """
    Trains the model using the optimizer for a single epoch.
    :param model: pytorch model
    :param optimizer:
    :return:
    """
    global total_train_step_count

    start_time = time.time()

    model.train()

    batch_idx = 1
    total_loss = 0

    for step, batch in enumerate(dataloader):

        x, y = batch["x"].to(device), batch["y"].squeeze(-1)

        out = model(x)

        y_pred1 = out.cpu()

        ce_loss = F.cross_entropy(y_pred1, y, weight=loss_weights1)
        if torch.isnan(torch.Tensor([ce_loss.item()])):
            print(f"pred : {y_pred1}")
            for k, v in batch.items():
                print(f"{k} : {v}")

        loss = ce_loss

        total_loss += loss.item()

        loss.backward()

        optimizer.step()

        torch_helper.show_progress(batch_idx, np.ceil(len(train_id_list) / batch_size), start_time,
                                   round(total_loss / (step + 1), 4))

        batch_idx += 1

        total_train_step_count += 1
        if total_train_step_count % 10 == 0:
            if logger_type == "tensorboard":
                summary.add_scalar(
                    "train_ce_loss", ce_loss.cpu().item(), total_train_step_count)
                summary.add_scalar("train_total_loss",
                                   loss.cpu().item(), total_train_step_count)
            if logger_type == "wandb":
                wandb.log({"train_step/ce_loss": ce_loss.cpu().item(),
                          "train_step": total_train_step_count})

    return model


# ----------------------------------------------------------------------------
# Evaluate the model
# ----------------------------------------------------------------------------
def evaluate(model, dataloader, device, epoch, logger_prefix):
    model.eval()

    total_loss1 = 0

    y1_true, y2_true, y3_true = [], [], []

    predictions = [[], [], []]

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader)):

            x, y = \
                batch["x"].to(device), batch["y"].squeeze(-1)

            out = model(x)

            y_pred1 = out.cpu()

            predictions[0].extend(list(torch.argmax(y_pred1, -1).numpy()))
            y1_true.extend(list(y.numpy()))

            loss2 = F.cross_entropy(y_pred1, y)
            total_loss1 += loss2.item()

    micro_f1_2 = f1_score(y1_true, predictions[0], average='weighted')
    average_type = ["micro", "macro", "weighted"]
    f1 = {k: f1_score(y1_true, predictions[0], average=k)
          for k in average_type}

    precision = {k: precision_score(
        y1_true, predictions[0], average=k) for k in average_type}

    recall = {k: recall_score(
        y1_true, predictions[0], average=k) for k in average_type}

    for t in average_type:
        print("****************")
        print(f"f1 {t} : {f1[t]}")
        print(f"precision {t} : {precision[t]}")
        print(f"recall {t} : {recall[t]}")

    print("****************")
    if logger_type == "wandb":
        for t in average_type:
            wandb.log({
                f"{logger_prefix}_score_{t}/f1": f1[t],
                f"{logger_prefix}_score_{t}/recall": recall[t],
                f"{logger_prefix}_score_{t}/precision": precision[t],
                "epoch": epoch

            })

    return predictions, \
        total_loss1 / len(dataloader), \
        micro_f1_2


def training_loop(batch_size, device):
    """

    :return:
    """

    print("started loading datalodaers...")
    train_dataloader = get_dataloader("train", batch_size, base_dir)
    val_dataloader = get_dataloader("val", batch_size, base_dir)
    test_dataloader = get_dataloader("test", batch_size, base_dir)
    print("successfully loaded dataloaders.")

    model = create_model(device)

    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(
        ), lr=learning_rate, momentum=0.9, weight_decay=weight_decay_val)

    lr_scheduler = reduce_on_plateau_lr_schdlr(
        optimizer, 'max', min_lr=1e-8, patience=2, factor=0.5)

    for epoch in range(start_epoch, max_epochs):
        print('[Epoch %d] / %d : %s' % (epoch + 1, max_epochs, run_name))

        model = train(model, optimizer, train_dataloader, device)

        val_pred, val_loss1, val_f1 = evaluate(
            model, val_dataloader, device, epoch, "val")
        train_pred, train_loss1, train_f1 = evaluate(
            model, train_dataloader, device, epoch, "train")
        if logger_type == "tensorboard":
            summary.add_scalar("train_loss_epoch", train_loss1, epoch)
            summary.add_scalar("train_f1_epoch", train_f1, epoch)
            summary.add_scalar("val_loss", val_loss1, epoch)
            summary.add_scalar("val_f1", val_f1, epoch)

        if logger_type == "wandb":
            wandb.log({
                "train_epoch/loss": train_loss1,
                "val_epoch/loss": val_loss1,
                "epoch": epoch
            })

        current_lr = 0
        for pg in optimizer.param_groups:
            current_lr = pg['lr']

        print('Validation Loss %.5f, Validation F1 %.5f' % (val_loss1, val_f1))

        print('Learning Rate', current_lr)

        if lr_schedule_active:
            lr_scheduler.step(val_f1)

        is_best = torch_helper.checkpoint_model(model, optimizer, output_dir_path, val_f1, epoch + 1,
                                                'max')

        print()

        # -------------------------------------------------------------
        # Tensorboard Logging
        # -------------------------------------------------------------
        info = {'training loss': train_loss1,
                'validation loss': val_loss1,
                'train_f1_1': train_f1,
                'val_f1_1': val_f1,
                'lr': current_lr
                }


def test(batch_size, device):
    model = create_model(device)
    test_dataloader = get_dataloader("test", batch_size, base_dir)
    val_pred, val_loss1, val_f1 = evaluate(
        model, test_dataloader, device, 0, "test")
    print('Validation Loss %.5f, Validation F1 %.5f' % (val_loss1, val_f1))


if __name__ == '__main__':

    if run_mode == "test":
        test(batch_size, device)
    else:
        with open(output_dir_path + 'description.txt', 'w') as f:
            f.write(description)
            f.close()

        training_loop(batch_size, device)
