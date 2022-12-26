import torch
from core.config import cfg, update_cfg
from core.train_helper import run
from core.get_data import create_dataset
from core.get_model import create_model


def train(train_loader, model, optimizer, evaluator, device, sharp):
    y_preds, y_trues = [], []
    total_loss = 0
    N = 0
    criterion = torch.nn.CrossEntropyLoss()
    for data in train_loader:
        if model.use_lap:
            batch_pos_enc = data.lap_pos_enc
            sign_flip = torch.rand(batch_pos_enc.size(1))
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            data.lap_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        N += data.num_graphs
        y_preds.append(torch.argmax(out, dim=-1))
        y_trues.append(data.y)
        optimizer.step()
    y_preds = torch.cat(y_preds, -1)
    y_trues = torch.cat(y_trues, -1)
    perm = (y_preds == y_trues).float().mean()
    loss = total_loss / N
    return perm, loss


@ torch.no_grad()
def test(loader, model, evaluator, device):
    y_preds, y_trues = [], []
    total_loss = 0
    N = 0
    criterion = torch.nn.CrossEntropyLoss()
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y)
        total_loss += loss.item() * data.num_graphs
        N += data.num_graphs
        y_preds.append(torch.argmax(out, dim=-1))
        y_trues.append(data.y)
    y_preds = torch.cat(y_preds, -1)
    y_trues = torch.cat(y_trues, -1)
    perm = (y_preds == y_trues).float().mean()
    loss = total_loss / N
    return perm, loss


if __name__ == '__main__':
    # get config
    cfg.merge_from_file('train/configs/GraphMLPMixer/mnist.yaml')
    cfg = update_cfg(cfg)
    run(cfg, create_dataset, create_model, train, test)
