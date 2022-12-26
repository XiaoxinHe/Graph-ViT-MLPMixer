import torch
from core.config import cfg, update_cfg
from core.train_helper import run
from core.get_data import create_dataset
from core.get_model import create_model


def train(train_loader, model, optimizer, evaluator, device, sharp):
    total_loss = 0
    N = 0
    for data in train_loader:
        if model.use_lap:
            batch_pos_enc = data.lap_pos_enc
            sign_flip = torch.rand(batch_pos_enc.size(1))
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            data.lap_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
        data, y, num_graphs = data.to(device), data.y, data.num_graphs
        optimizer.zero_grad()
        loss = (model(data).squeeze() - y).abs().mean()
        loss.backward()
        total_loss += loss.item() * num_graphs
        optimizer.step()
        N += num_graphs
    train_loss = total_loss/N
    train_perf = - train_loss
    return train_loss, train_perf


@ torch.no_grad()
def test(loader, model, evaluator, device):
    total_loss = 0
    N = 0
    for data in loader:
        data, y, num_graphs = data.to(device), data.y, data.num_graphs
        loss = (model(data).squeeze() - y).abs().mean()
        total_loss += loss.item() * num_graphs
        N += num_graphs
    test_loss = total_loss / N
    test_perf = - test_loss
    return test_perf, test_loss


if __name__ == '__main__':
    cfg.merge_from_file('train/configs/GraphMLPMixer/zinc.yaml')
    cfg = update_cfg(cfg)
    run(cfg, create_dataset, create_model, train, test)
