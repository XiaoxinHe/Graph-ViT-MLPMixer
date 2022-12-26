import torch
import numpy as np
from core.config import cfg, update_cfg
from core.train_helper import run
from core.get_data import create_dataset
from core.get_model import create_model
from sklearn.metrics import average_precision_score


def eval_ap(y_true, y_pred):
    '''
        compute Average Precision (AP) averaged across tasks
    '''

    ap_list = []
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            ap = average_precision_score(y_true[is_labeled, i],
                                         y_pred[is_labeled, i])

            ap_list.append(ap)

    if len(ap_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute Average Precision.')

    return sum(ap_list) / len(ap_list)


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
        optimizer.zero_grad()
        mask = ~torch.isnan(data.y)
        y = data.y.to(torch.float)
        out = model(data)
        y_preds.append(out)
        y_trues.append(y)
        loss = criterion(out[mask], y[mask])
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        N += data.num_graphs
        optimizer.step()

    y_preds = torch.cat(y_preds, dim=0)
    y_trues = torch.cat(y_trues, dim=0)

    train_perf = eval_ap(y_true=y_trues, y_pred=y_preds)
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

    y_preds = torch.cat(y_preds, dim=0)
    y_trues = torch.cat(y_trues, dim=0)

    test_perf = eval_ap(y_true=y_trues, y_pred=y_preds)
    test_loss = total_loss/N
    return test_perf, test_loss


if __name__ == '__main__':
    cfg.merge_from_file('train/configs/GraphMLPMixer/peptides_func.yaml')
    cfg = update_cfg(cfg)
    run(cfg, create_dataset, create_model, train, test, evaluator=None)
