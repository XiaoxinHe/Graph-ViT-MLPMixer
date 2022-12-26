import torch
from core.config import cfg, update_cfg
from core.train_helper import run
from core.get_data import create_dataset
from core.get_model import create_model


def train(train_loader, model, optimizer, evaluator, device, sharp):
    total_loss = 0
    total_correct = 0
    N = 0
    criterion = torch.nn.functional.cross_entropy
    for data in train_loader:
        if model.use_lap:
            batch_pos_enc = data.lap_pos_enc
            sign_flip = torch.rand(batch_pos_enc.size(1))
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            data.lap_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
        data, y, num_graphs = data.to(device), data.y, data.num_graphs
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(input=out, target=data.y)
        loss.backward()
        total_loss += loss.item() * num_graphs
        _, train_pred = out.max(dim=1)
        total_correct += train_pred.eq(data.y).sum().item()
        optimizer.step()
        N += num_graphs
    train_perf = total_correct/N
    train_loss = total_loss/N
    return train_perf, train_loss


@ torch.no_grad()
def test(loader, model, evaluator, device):
    total_loss = 0
    total_correct = 0
    N = 0
    criterion = torch.nn.functional.cross_entropy
    for data in loader:
        data, y, num_graphs = data.to(device), data.y, data.num_graphs
        out = model(data)
        loss = criterion(input=out, target=y)
        total_loss += loss.item() * num_graphs
        _, pred = out.max(dim=1)
        total_correct += pred.eq(y).sum().item()
        N += num_graphs
    test_loss = total_loss / N
    test_perf = total_correct/N
    return test_perf, test_loss


if __name__ == '__main__':
    cfg.merge_from_file('train/configs/GraphMLPMixer/tree_neighbour.yaml')
    cfg = update_cfg(cfg)
    run(cfg, create_dataset, create_model, train, test)
