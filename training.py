import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import sacred
import torch
import torch.cuda.amp
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import utils
from dataloading import SubsetWithTransform
from evaluation import evaluate
from experiment import ex, setup_model, setup_optimizer, setup_transforms, \
    get_dataset
from models import cp_lossfn


class Metrics:
    @ex.capture
    def __init__(self, n_inner_splits, n_outer_splits,
                 break_after_first_inner_split, break_after_first_outer_split,
                 early_stopping_metrics):
        self.n_inner = n_inner_splits if not break_after_first_inner_split else 1
        self.n_outer = n_outer_splits if not break_after_first_outer_split else 1
        self.acc = {split: np.zeros((self.n_outer, self.n_inner)) for split in
                    ["xval", "test"]}
        self.auc = {split: np.zeros((self.n_outer, self.n_inner)) for split in
                    ["xval", "test"]}
        self.bal_acc = {split: np.zeros((self.n_outer, self.n_inner)) for split
                        in ["xval", "test"]}
        self.loss = {split: np.zeros((self.n_outer, self.n_inner)) for split in ["train", "xval", "test"]}
        self.best_epoch = {metric: np.zeros((self.n_outer, self.n_inner)) for metric in early_stopping_metrics}

    def record(self, k1, k2, idx1, idx2, v):
        getattr(self, k1)[k2][idx1, idx2] = v

    @ex.capture
    def report(self, _log, num_classes):
        for outer_fold_idx in range(self.n_outer):
            _log.info(f"Outer fold {outer_fold_idx}:")
            metrics_to_report = ["acc", "bal_acc", "loss"]
            if num_classes == 2:
                metrics_to_report.append("auc")
            for metric in metrics_to_report:
                xval_mean = getattr(self, metric)["xval"][outer_fold_idx].mean()
                xval_std = getattr(self, metric)["xval"][outer_fold_idx].std()
                test_mean = getattr(self, metric)["test"][outer_fold_idx].mean()
                test_std = getattr(self, metric)["test"][outer_fold_idx].std()
                _log.info(f"    {metric} xval/test: {xval_mean:.4f} (+-{xval_std:.4f}) / {test_mean:.4f} (+-{test_std:.4f})")
            for metric in self.best_epoch:
                best_epoch_mean = self.best_epoch[metric][outer_fold_idx].mean()
                best_epoch_std = self.best_epoch[metric][outer_fold_idx].std()
                _log.info(f"    {metric} best epoch: {best_epoch_mean} (+-{best_epoch_std})")


@ex.capture
def run_train_nn(n_outer_splits, num_workers,
                 batch_size,
                 break_after_first_inner_split, break_after_first_outer_split,
                 reweight_classes,
                 num_classes, log_dir_root, snapshot,
                 split_file, subsample, group_by_case_in_val,
                 _seed, _log, _run):
    torch.manual_seed(_seed)
    print('batch_size: {}'.format(batch_size))


    #device = "cuda" if torch.cuda.is_available() else "cpu"

    ex.info["process_id"] = os.getpid()
    if snapshot:
        log_dir = Path(snapshot).parent.parent.parent
    else:
        log_dir = Path(log_dir_root) / str(_run._id)
    splits = ["train", "val", "test"]

    # Setup data loading
    data_transforms = setup_transforms()
    dataset = get_dataset()
    classes = dataset.classes

    print("Class to index mapping:")
    for idx, class_name in enumerate(classes):
        print(f"{idx}: {class_name}")

    _run.info["classes"] = classes
    _log.info(f'Loaded {len(dataset)} images.')

    # Setup model and training
    model = setup_model()

    if torch.cuda.device_count() > 1:
        print(f"✅ Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Store initial params for model reset in each fold
    if snapshot:
        # Load from disk
        initial_params = torch.load(snapshot, map_location=device)
    else:
        initial_params = copy.deepcopy(model.state_dict())

    with open(split_file) as f:
        cases = json.load(f)
        print("Cases: {}".format(cases))
        indices = {split:
                       [i for i in range(len(dataset.imgs)) if
                        str(dataset.groups[i]) in str(cases[split])]
                   for split in splits
                   }

    from collections import Counter

    # Print full dataset class distribution before any split
    all_targets = dataset.targets
    print("📊 Full dataset class distribution (before any train/val/test split):")
    print(Counter(all_targets))

    # Check class distribution in 'train' portion of split_file
    train_targets_full = [dataset.targets[i] for i in indices['train']]
    print("✅ Full training set class distribution before any internal split:")
    print(Counter(train_targets_full))

    outer_splits = [[indices[split] for split in splits]]
    metrics = Metrics()

    for split in outer_splits:
        assert len(split) > 0

    for outer_fold_idx, (idx_train, idx_xval, idx_test) in enumerate(
            outer_splits):
        inner_fold_idx = 0  # Nested crossvalidation not implemented yet
        _log.info(f"CV fold outer {outer_fold_idx}/{n_outer_splits}")
        _log.info(f"Train set size: {len(idx_train)}")
        _log.info(f"Val set size: {len(idx_xval)}")
        _log.info(f"Test set size: {len(idx_test)}")
        test_set = SubsetWithTransform(dataset, idx_test,
                                       data_transforms=data_transforms['test'])
        test_loader = DataLoader(test_set, batch_size=batch_size,
                                 num_workers=num_workers, drop_last=False)

        # Reset random seeds and network parameters
        sacred.randomness.set_global_seed(_seed)
        torch.manual_seed(_seed)
        model.load_state_dict(initial_params)

        # Setup logging
        log_dir.mkdir(exist_ok=True, parents=True)

        # Data loading
        if subsample is not None:
            len_before = len(idx_train)
            idx_train = dataset.subsample(idx_train, subsample)
            print(f"Subsampled trainset from {len_before} to {len(idx_train)}")

        if not group_by_case_in_val:
            # Destroy case grouping
            idx_dev = idx_train + idx_xval
            ys = [dataset.targets[i] for i in idx_dev]

            print("✅ Class distribution BEFORE stratified split (train + val):")
            print(Counter(ys))

            idx_train, idx_xval = train_test_split(idx_dev, test_size=0.3,
                                                   stratify=ys)

            train_labels = [dataset.targets[i] for i in idx_train]
            print("✅ Class distribution in TRAIN set AFTER stratified split:")
            print(Counter(train_labels))

        train_set = SubsetWithTransform(dataset, idx_train,
                                        data_transforms=data_transforms[
                                            'train'])
        xval_set = SubsetWithTransform(dataset, idx_xval,
                                       data_transforms=data_transforms['xval'])
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers, drop_last=True)
        xval_loader = DataLoader(xval_set, batch_size=batch_size,
                                 num_workers=num_workers, drop_last=False)
        _log.info(
            f"Loaded {len(train_set)} train and {len(xval_set)} xval samples.")

        # Inner fold training
        class_weights = []
        if reweight_classes:
            train_targets = [dataset.targets[i] for i in train_set.indices]

        #    print('train_targets: ', train_targets, num_classes)
        #    class_weights = [len(train_targets) / train_targets.count(c) for c
        #                     in range(num_classes)]

            for c in range(num_classes):
                count = train_targets.count(c)
                print(f'Class {c} : {count}')
                if count == 0:
                    print(f"⚠️ Warning: Class {c} is missing in the training set!")
                    weight = 0  # or float('inf') or some small value like 1e-8
                else:
                    weight = len(train_targets) / count
                class_weights.append(weight)

            _log.info("Class weights")
            #for i, c in enumerate(class_weights):
            #    _log.info(f"{classes[i]}: {c:.4f}")
        else:
            _log.info("Do not reweight classes.")

        if reweight_classes:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        else:
            class_weights_tensor = None

        #criterion = nn.CrossEntropyLoss(reduction="sum",
        #                                weight=torch.Tensor(
        #                                    class_weights).to(
        #                                    device) if reweight_classes else None)

        criterion = nn.CrossEntropyLoss(reduction="sum", weight=class_weights_tensor)
        if snapshot:
            _log.info(
                "Load model from snapshot => no training, only evaluation.")
            return_embedding = True
        else:
            model, train_losses, best_epoch = train_inner_fold(model,
                                                               train_loader,
                                                               xval_loader,
                                                               criterion,
                                                               log_dir,
                                                               outer_fold_idx,
                                                               0,
                                                               classes)

            # Inner fold evaluation and logging
            metrics.record("loss", "train", outer_fold_idx, inner_fold_idx,
                           sum(train_losses) / len(train_losses))
            for metric, epoch in best_epoch.items():
                metrics.record("best_epoch", metric, outer_fold_idx,
                               inner_fold_idx, epoch)
            return_embedding = False

        for split, loader in [
            ("train", train_loader),
            ("xval", xval_loader),
            ("test", test_loader)
        ]:
            scores, y_true, y_score, embeddings, paths = evaluate(model, loader,
                                                                  criterion,
                                                                  classes,
                                                                  return_embedding=return_embedding,
                                                                  return_prediction=True)
            if split != "train":
                for metric, value in scores.items():
                    metrics.record(metric, split, outer_fold_idx,
                                   inner_fold_idx, value)

            df = utils.setup_prediction_df(y_score, y_true, paths, classes,
                                           suffix=f"-fold{inner_fold_idx}")
            csv_file = Path(log_dir) / f"predictions_{split}.csv"
            if not csv_file.exists():
                df.to_csv(csv_file)

            if len(embeddings) > 0:
                np.save(Path(log_dir) / f"embeddings-full_{split}.npy", embeddings)

        if break_after_first_inner_split:
            break

        if break_after_first_outer_split:
            break


    metrics.report()
    result = {"bal_acc": metrics.bal_acc["test"].mean(),
              "loss": metrics.loss["test"].mean()}

    return result


@ex.capture
def train_inner_fold(model, train_loader, xval_loader, criterion, log_dir,
                     outer_fold_idx, inner_fold_idx,
                     classes, num_epochs, xval_interval, early_stopping_metrics,
                     patience, _log):

    print('Total Epochs:', num_epochs)

    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = next(model.parameters()).device
    model.to(device)
    _log.info(
        f"STARTING outer fold {outer_fold_idx} inner fold {inner_fold_idx}")
    # Reinitialize for every fold to reset momentum, LR decay etc.

    if criterion.weight is not None:
        criterion.weight = criterion.weight.to(device)

    optimizer, scheduler = setup_optimizer(model)
    train_losses = []
    best_performance = {metric: -float("inf") for metric in
                        early_stopping_metrics}
    epochs_not_improved = {metric: 0 for metric in early_stopping_metrics}
    best_epoch = {metric: -1 for metric in early_stopping_metrics}
    #model = model.to(device)

    # Mixed precison scaler
    scaler = torch.cuda.amp.GradScaler()

    writer = SummaryWriter(log_dir=os.path.join(log_dir, f"tensorboard-outer-fold{outer_fold_idx}-inner-fold{inner_fold_idx}"))

    # Main training procedure
    for epoch in range(num_epochs):
        start_time = time.time()

        epoch_losses = train_epoch(model, train_loader, criterion, optimizer,
                                   scheduler, scaler)
        train_losses.extend(epoch_losses)
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

        ex.log_scalar(
            f"train_loss-outer_fold{outer_fold_idx}-inner_fold{inner_fold_idx}",
            avg_loss, step=epoch)
        writer.add_scalar('Loss/train', avg_loss, epoch)

        for i, epoch_loss in enumerate(epoch_losses):
            ex.log_scalar(
                f"train_loss_iteration-outer_fold{outer_fold_idx}-inner_fold{inner_fold_idx}",
                epoch_loss, step=epoch + i / len(epoch_losses))

        # Validation
        if epoch % xval_interval == 0:
            curr_xval_scores = evaluate(model, xval_loader, criterion, classes)

            # Log validation scores
            for k,v in curr_xval_scores.items():
                if not k == "conf_mat":
                    ex.log_scalar(f"xval_{k}-outer_fold{outer_fold_idx}-inner_fold{inner_fold_idx}",
                                  curr_xval_scores[k], step=epoch)
                    writer.add_scalar(f'Validation/{k}', v, epoch)

            # Check improvement and early stopping
            for early_stopping_metric in early_stopping_metrics:
                xval_performance = curr_xval_scores[early_stopping_metric]
                if early_stopping_metric == "loss":
                    # Lower loss is better
                    xval_performance *= -1
                if xval_performance > best_performance[early_stopping_metric]:
                    best_performance[early_stopping_metric] = xval_performance
                    best_epoch[early_stopping_metric] = epoch
                    epochs_not_improved[early_stopping_metric] = 0
                    torch.save(model.state_dict(), os.path.join(log_dir,
                                                                f"model-{early_stopping_metric}.pt"))
                    _log.info(
                        f"Validation {early_stopping_metric} increased to {xval_performance:.4f} in epoch {epoch} "
                        f"for outer fold {outer_fold_idx}, inner fold {inner_fold_idx}.")
                else:
                    epochs_not_improved[early_stopping_metric] += 1
            if min(epochs_not_improved.values()) > patience:
                _log.info(f"Early stopping after {epoch} epochs.")
                break
        print(f"Epoch {epoch} took {time.time() - start_time:.2f}s")
    _log.info(f"FINISHED outer fold {outer_fold_idx} inner fold {inner_fold_idx}")

    # Reset to the best parameters in this run (according to the main metric)
    single_metric = "bal_acc" if len(early_stopping_metrics) > 1 else early_stopping_metrics[0]
    model.load_state_dict(torch.load(os.path.join(log_dir, f"model-{single_metric}.pt")))
    writer.close()
    return model, train_losses, best_epoch


@ex.capture
def train_epoch(model, dataloader, criterion, optimizer, scheduler, scaler,
                cp_weight):

    print('cp_weight: {}'.format(cp_weight))
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    torch.cuda.empty_cache()
    #model = model.to(device)

    model.train()
    if criterion.weight is not None:
        criterion.weight = criterion.weight.to(device)
    losses = list()


    for batch_idx, (data, target, path) in tqdm(enumerate(dataloader),
                                                total=len(dataloader)):

        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(data)  # NxTxK

            # Consistency prior
            if cp_weight is not None and cp_weight > 0:
                loss = criterion(output[:, 0],
                                 target)  # Only use untransformed version here
                loss_cp = cp_lossfn(output)
            else:
                loss = criterion(output, target)
                loss_cp = torch.Tensor([0.0]).to(model.device)
            loss = cp_weight * loss_cp + (1 - cp_weight) * loss

        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        losses.append(loss.item() / len(data))  # Average loss in batch
        torch.cuda.empty_cache()
    scheduler.step()
    return losses
