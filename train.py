import os
import random
import argparse
import json
import logging
import time
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

from model.batch import generate_batch
from model.model import PetLoverCenter
from model.config import PetLoverCenterConfig
from data_loader.data_loader import PetDataset
# from model.loss import loss_function
from utils.utils import plot_results

from sklearn.metrics import mean_squared_error


parser = argparse.ArgumentParser(
    description="Predict the “Pawpularity” of pet photos")
parser.add_argument('--config_path', help='path to config file')


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def train(data_loader, model, criterion, optimizer, device):
    model.train()
    losses = 0.0

    for i, batch in enumerate(data_loader):
        batch.to_device(device)
        optimizer.zero_grad()
        # out, mu, logvar = model(batch.imgs, batch.xs, True)
        # loss = criterion(out, batch.ys, mu, logvar)
        out = model(batch.imgs, batch.xs, True)
        loss = criterion(out, torch.unsqueeze(batch.ys, 1))
        losses += loss.item()
        loss.backward()
        optimizer.step()

    return losses/len(data_loader)


def test(data_loader, model, criterion, device):
    model.eval()
    losses = 0.0
    mse_losses = 0.0

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            batch.to_device(device)
            # out, mu, logvar = model(batch.imgs, batch.xs, False)
            # loss = criterion(out, batch.ys, mu, logvar)
            out = model(batch.imgs, batch.xs, True)
            loss = criterion(out, torch.unsqueeze(batch.ys, 1))
            losses += loss.item()
            if device == "cuda":
                mse_losses += mean_squared_error(
                    out.cpu().numpy(), batch.ys.cpu().numpy())
            else:
                mse_losses += mean_squared_error(out, batch.ys)

    return losses/len(data_loader), 100 * mse_losses/len(data_loader)


if __name__ == '__main__':
    args = parser.parse_args()
    config = PetLoverCenterConfig(json.load(open(args.config_path)))

    set_seed(config.seed)
    data_dir = os.path.join(config.data_dir, "train.csv")
    img_dir = os.path.join(config.data_dir, "train")
    data = PetDataset(data_dir, img_dir, config.img_size, transform=None)

    num_data = len(data)
    num_val = round(num_data * config.split_ratio)
    num_train = num_data - num_val

    train_data, val_data = random_split(data, [num_train, num_val])

    tr_data_loader = DataLoader(
        train_data, batch_size=config.tr_batch_size,
        shuffle=True, collate_fn=generate_batch)
    te_data_loader = DataLoader(
        val_data, batch_size=config.val_batch_size,
        shuffle=False, collate_fn=generate_batch)

    save_model_dir = os.path.join(config.save_dir, 'model')
    log_dir = os.path.join(config.save_dir, 'log')
    config_dir = os.path.join(config.save_dir, 'config')

    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    logging.basicConfig(
        filename=os.path.join(log_dir, 'train.log'), level=logging.INFO)
    writer = SummaryWriter(log_dir)
    writer.add_text(
        'training_config', json.dumps(json.load(open(args.config_path))))
    with open(os.path.join(config_dir, 'config.json'), 'w') as f:
        json.dump(json.load(open(args.config_path)), f, indent=2)

    device = config.device
    model = PetLoverCenter(config, logging)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # criterion = loss_function
    criterion = nn.MSELoss()
    epochs = config.epochs

    logging.info('Begin training')
    logging.info(config.msg)
    logging.info('Training size: {} | Validation size: {}'.format(
        num_train, num_val))
    print('Begin training')
    print('Training size: {} | Validation size: {}'.format(num_train, num_val))

    best_mse = 100
    try:
        tr_losses = []
        val_losses = []
        val_mse_losses = []
        for epoch in range(1, epochs + 1):
            start = time.time()
            tr_loss = train(
                tr_data_loader, model, criterion, optimizer, device)
            val_loss, val_mse_loss = test(
                te_data_loader, model, criterion, device)

            writer.add_scalars(
                'losses', {'training': tr_loss, 'validation': val_loss}, epoch)
            tr_losses.append(tr_loss)
            val_losses.append(val_loss)
            val_mse_losses.append(val_mse_loss)

            if val_mse_loss < best_mse:
                with open(os.path.join(save_model_dir, "model.pt"), 'wb') as f:
                    torch.save(model.state_dict(), f)
                best_mse = val_mse_loss

            logging.info('| epoch {:3d} | time {:3.2f} mins | tr_loss {:1.3f}'
                         ' | val_loss {:1.3f} | val_mse_acc {:1.3f} | '
                         'best_mse {:1.3f} |'.format(
                             epoch, (time.time() - start) / 60, tr_loss,
                             val_loss, val_mse_loss, best_mse))
            print('| epoch {:3d} | time {:3.2f} mins | tr_loss {:1.3f}'
                  ' | val_loss {:1.3f} | val_mse_acc {:1.3f} | '
                  'best_mse {:1.3f} |'.format(
                     epoch, (time.time() - start) / 60, tr_loss,
                     val_loss, val_mse_loss, best_mse))

        plot_results(log_dir, 'loss', tr_losses, val_losses)
        writer.close()
    except KeyboardInterrupt:
        writer.close()
        print('Exiting from training early')
