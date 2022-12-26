import os
import torch
import random
import time
import numpy as np
from core.log import config_logger
from core.asam import ASAM
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(cfg, create_dataset, create_model, train, test, evaluator=None):
    if cfg.seed is not None:
        seeds = [cfg.seed]
        cfg.train.runs = 1
    else:
        seeds = [41, 95, 12, 35]

    writer, logger = config_logger(cfg)

    train_dataset, val_dataset, test_dataset = create_dataset(cfg)

    train_loader = DataLoader(
        train_dataset, cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(
        val_dataset,  cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(
        test_dataset, cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)

    train_losses = []
    train_perfs = []
    vali_perfs = []
    test_perfs = []
    per_epoch_times = []
    total_times = []
    for run in range(cfg.train.runs):
        set_seed(seeds[run])
        model = create_model(cfg).to(cfg.device)
        print(f"\nNumber of parameters: {count_parameters(model)}")

        if cfg.train.optimizer == 'ASAM':
            sharp = True
            optimizer = torch.optim.SGD(
                model.parameters(), lr=cfg.train.lr, momentum=0.9, weight_decay=cfg.train.wd)
            minimizer = ASAM(optimizer, model, rho=0.5)

        else:
            sharp = False
            optimizer = torch.optim.Adam(
                model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=cfg.train.lr_decay,
                                                               patience=cfg.train.lr_patience,
                                                               verbose=True)

        start_outer = time.time()
        per_epoch_time = []
        train_perf = best_val_perf = test_perf = float('-inf')
        for epoch in range(cfg.train.epochs):
            start = time.time()
            model.train()
            train_perf, train_loss = train(
                train_loader, model, optimizer if not sharp else minimizer, evaluator=evaluator, device=cfg.device, sharp=sharp)
            model.eval()
            val_perf, val_loss = test(val_loader, model,
                                      evaluator=evaluator, device=cfg.device)
            if val_perf >= best_val_perf:
                best_val_perf = val_perf
                test_perf, test_loss = test(test_loader, model,
                                            evaluator=evaluator, device=cfg.device)

            time_cur_epoch = time.time() - start
            per_epoch_time.append(time_cur_epoch)

            # memory_allocated = torch.cuda.max_memory_allocated(
            #     cfg.device) // (1024 ** 2)
            # memory_reserved = torch.cuda.max_memory_reserved(
            #     cfg.device) // (1024 ** 2)

            print(f'Epoch: {epoch:03d}, Train perf: {train_perf:.4f}, Train Loss: {train_loss:.4f}, '
                  f'Val: {val_perf:.4f}, Test: {test_perf:.4f}, Seconds: {time_cur_epoch:.4f}')
            #   f'Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')

            writer.add_scalar(f'Run{run}/train-loss', train_loss, epoch)
            writer.add_scalar(f'Run{run}/train-perf', train_perf, epoch)
            writer.add_scalar(f'Run{run}/val-loss', val_loss, epoch)
            writer.add_scalar(f'Run{run}/val-perf', val_perf, epoch)
            writer.add_scalar(f'Run{run}/test-loss', test_loss, epoch)
            writer.add_scalar(f'Run{run}/test-perf', test_perf, epoch)

            if scheduler is not None:
                scheduler.step(val_loss)

            if not sharp:
                if optimizer.param_groups[0]['lr'] < cfg.train.min_lr:
                    print("!! LR EQUAL TO MIN LR SET.")
                    break

            if cfg.dataset in ['TreeDataset', 'sr25-classify'] and test_perf == 1.0:
                break
            # torch.cuda.empty_cache()  # empty test part memory cost

        per_epoch_time = np.mean(per_epoch_time)
        total_time = (time.time()-start_outer)/3600

        print("\nRun: ", run)
        print("Train Loss: {:.4f}".format(train_loss))
        print("Train Accuracy: {:.4f}".format(train_perf))
        print("Vali Accuracy: {:.4f}".format(best_val_perf))
        print("Test Accuracy: {:.4f}".format(test_perf))
        print("Convergence Time (Epochs): {}".format(epoch+1))
        print("AVG TIME PER EPOCH: {:.4f} s".format(per_epoch_time))
        print("TOTAL TIME TAKEN: {:.4f} h".format(total_time))

        train_losses.append(train_loss)
        train_perfs.append(train_perf)
        vali_perfs.append(best_val_perf)
        test_perfs.append(test_perf)
        per_epoch_times.append(per_epoch_time)
        total_times.append(total_time)

    if cfg.train.runs > 1:
        train_loss = torch.tensor(train_losses)
        train_perf = torch.tensor(train_perfs)
        test_perf = torch.tensor(test_perfs)
        vali_perf = torch.tensor(vali_perfs)
        per_epoch_time = torch.tensor(per_epoch_times)
        total_time = torch.tensor(total_times)
        print(f'\nFinal Train Loss: {train_loss.mean():.4f} ± {train_loss.std():.4f}'
              f'\nFinal Train: {train_perf.mean():.4f} ± {train_perf.std():.4f}'
              f'\nFinal Vali: {vali_perf.mean():.4f} ± {vali_perf.std():.4f}'
              f'\nFinal Test: {test_perf.mean():.4f} ± {test_perf.std():.4f}'
              f'\nSeconds/epoch: {per_epoch_time.mean():.4f}'
              f'\nHours/total: {total_time.mean():.4f}')
        logger.info("-"*50)
        logger.info(cfg)
        logger.info(f'\nFinal Train Loss: {train_loss.mean():.4f} ± {train_loss.std():.4f}'
                    f'\nFinal Train: {train_perf.mean():.4f} ± {train_perf.std():.4f}'
                    f'\nFinal Vali: {vali_perf.mean():.4f} ± {vali_perf.std():.4f}'
                    f'\nFinal Test: {test_perf.mean():.4f} ± {test_perf.std():.4f}'
                    f'\nSeconds/epoch: {per_epoch_time.mean():.4f}'
                    f'\nHours/total: {total_time.mean():.4f}')


def count_parameters(model):
    # For counting number of parameteres: need to remove unnecessary DiscreteEncoder, and other additional unused params
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def k_fold(dataset, folds=10):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)
    train_indices, test_indices = [], []
    ys = dataset.data.y
    for train, test in skf.split(torch.zeros(len(dataset)), ys):
        train_indices.append(torch.from_numpy(train).to(torch.long))
        test_indices.append(torch.from_numpy(test).to(torch.long))
    return train_indices, test_indices


def run_k_fold(cfg, create_dataset, create_model, train, test, evaluator=None, k=10):
    if cfg.seed is not None:
        seeds = [cfg.seed]
        cfg.train.runs = 1
    else:
        seeds = [41, 95, 12, 35]

    writer, logger = config_logger(cfg)
    dataset, transform, transform_eval = create_dataset(cfg)

    if hasattr(dataset, 'train_indices'):
        k_fold_indices = dataset.train_indices, dataset.test_indices
    else:
        k_fold_indices = k_fold(dataset, k)

    train_losses = []
    train_perfs = []
    test_perfs = []
    test_best_perfs = []
    per_epoch_times = []
    total_times = []
    for run in range(cfg.train.runs):
        set_seed(seeds[run])
        for fold, (train_idx, test_idx) in enumerate(zip(*k_fold_indices)):
            train_dataset = dataset[train_idx]
            test_dataset = dataset[test_idx]
            train_dataset.transform = transform
            test_dataset.transform = transform_eval
            test_dataset = [x for x in test_dataset]

            if not cfg.metis.online:
                train_dataset = [x for x in train_dataset]

            train_loader = DataLoader(
                train_dataset, cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers)
            test_loader = DataLoader(
                test_dataset,  cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)

            model = create_model(cfg).to(cfg.device)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                   factor=cfg.train.lr_decay,
                                                                   patience=cfg.train.lr_patience,
                                                                   verbose=True)

            start_outer = time.time()
            per_epoch_time = []
            train_perf = test_perf = best_test_perf = float('-inf')
            for epoch in range(cfg.train.epochs):
                start = time.time()
                model.train()
                train_perf, train_loss = train(
                    train_loader, model, optimizer, evaluator=evaluator, device=cfg.device)
                model.eval()
                test_perf, test_loss = test(
                    test_loader, model, evaluator=evaluator, device=cfg.device)

                best_test_perf = test_perf if test_perf > best_test_perf else best_test_perf
                scheduler.step(test_loss)
                time_cur_epoch = time.time() - start
                per_epoch_time.append(time_cur_epoch)

                print(f'Epoch/Fold: {epoch:03d}/{fold}, Train Loss: {train_loss:.4f}, Train: {train_perf:.4f}, '
                      f'Test:{test_perf:.4f}, Best-Test: {best_test_perf:.4f}, Seconds: {time_cur_epoch:.4f}, ')
                writer.add_scalar(f'Run{run}/train-loss', train_loss, epoch)
                writer.add_scalar(f'Run{run}/train-perf', train_perf, epoch)
                writer.add_scalar(f'Run{run}/test-loss', test_loss, epoch)
                writer.add_scalar(f'Run{run}/test-perf', test_perf, epoch)

                if optimizer.param_groups[0]['lr'] < cfg.train.min_lr:
                    print("!! LR EQUAL TO MIN LR SET.")
                    break

            per_epoch_time = np.mean(per_epoch_time)
            total_time = (time.time()-start_outer)/3600

            print(
                f'Fold {fold}, Test: {best_test_perf}, Seconds/epoch: {per_epoch_time}')
            train_losses.append(train_loss)
            train_perfs.append(train_perf)
            test_perfs.append(test_perf)
            test_best_perfs.append(best_test_perf)
            per_epoch_times.append(per_epoch_time)
            total_times.append(total_time)

        print("\nRun: ", run)
        print("Train Loss: {:.4f}".format(train_loss))
        print("Train Accuracy: {:.4f}".format(train_perf))
        print("Test Accuracy: {:.4f}".format(test_perf))
        print("Convergence Time (Epochs): {}".format(epoch+1))
        print("AVG TIME PER EPOCH: {:.4f} s".format(per_epoch_time))
        print("TOTAL TIME TAKEN: {:.4f} h".format(total_time))

    if cfg.train.runs > 1:
        train_loss = torch.tensor(train_losses)
        train_perf = torch.tensor(train_perfs)
        test_perf = torch.tensor(test_perfs)
        test_best_perf = torch.tensor(test_best_perfs)
        per_epoch_time = torch.tensor(per_epoch_times)
        total_time = torch.tensor(total_times)

        print(f'\nFinal Train Loss: {train_loss.mean():.4f} ± {train_loss.std():.4f}'
              f'\nFinal Train: {train_perf.mean():.4f} ± {train_perf.std():.4f}'
              f'\nFinal Test: {test_perf.mean():.4f} ± {test_perf.std():.4f}'
              f'\nFinal Test Best: {test_best_perf.mean():.4f} ± {test_best_perf.std():.4f}'
              f'\nSeconds/epoch: {per_epoch_time.mean():.4f}'
              f'\nHours/total: {total_time.mean():.4f}')
        logger.info("-"*50)
        logger.info(cfg)
        logger.info(f'\nFinal Train Loss: {train_loss.mean():.4f} ± {train_loss.std():.4f}'
                    f'\nFinal Train: {train_perf.mean():.4f} ± {train_perf.std():.4f}'
                    f'\nFinal Test: {test_perf.mean():.4f} ± {test_perf.std():.4f}'
                    f'\nSeconds/epoch: {per_epoch_time.mean():.4f}'
                    f'\nHours/total: {total_time.mean():.4f}')
