from core.config import cfg, update_cfg
from core.train_helper import run
from core.get_data import create_dataset
from core.get_model import create_model

import torch
from ogb.graphproppred import Evaluator


def train(train_loader, model, optimizer, evaluator, device, sharp):
    total_loss = 0
    N = 0
    y_preds, y_trues = [], []
    criterion = torch.nn.BCEWithLogitsLoss()
    for data in train_loader:
        if model.use_lap:
            batch_pos_enc = data.lap_pos_enc
            sign_flip = torch.rand(batch_pos_enc.size(1))
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            data.lap_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
        data = data.to(device)
        mask = ~torch.isnan(data.y)
        y = data.y.to(torch.float)
        if sharp:
            # Ascent
            out = model(data)
            loss = criterion(out[mask], y[mask])
            loss.backward()
            optimizer.ascent_step()
            # Descent
            out = model(data)
            y_preds.append(out)
            y_trues.append(y)
            loss = criterion(out[mask], y[mask])
            loss.backward()
            optimizer.descent_step()
        else:
            optimizer.zero_grad()
            out = model(data)
            y_preds.append(out)
            y_trues.append(y)
            loss = criterion(out[mask], y[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * data.num_graphs
        N += data.num_graphs

    train_perf = evaluator.eval({
        'y_pred': torch.cat(y_preds, dim=0),
        'y_true': torch.cat(y_trues, dim=0),
    })[evaluator.eval_metric]
    train_loss = total_loss / N
    return train_perf, train_loss


@torch.no_grad()
def test(loader, model, evaluator, device):
    total_loss = 0
    N = 0
    y_preds, y_trues = [], []
    criterion = torch.nn.BCEWithLogitsLoss()
    for data in loader:
        data = data.to(device)
        mask = ~torch.isnan(data.y)
        y = data.y.to(torch.float)
        out = model(data)
        y_preds.append(out)
        y_trues.append(y)
        loss = criterion(out[mask], y[mask])
        total_loss += loss.item() * data.num_graphs
        N += data.num_graphs
    test_perf = evaluator.eval({
        'y_pred': torch.cat(y_preds, dim=0),
        'y_true': torch.cat(y_trues, dim=0),
    })[evaluator.eval_metric]
    test_loss = total_loss/N
    return test_perf, test_loss


if __name__ == '__main__':
    # get config
    cfg.merge_from_file('train/configs/GraphMLPMixer/molhiv.yaml')
    cfg = update_cfg(cfg)
    evaluator = Evaluator(cfg.dataset)
    run(cfg, create_dataset, create_model, train, test, evaluator=evaluator)
