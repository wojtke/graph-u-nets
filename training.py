import numpy as np
import optuna
import torch


class Trainer:
    def __init__(self, model, optimizer, criterion, device, writer=None, verbose=False):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.epoch = 0
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        self.writer = writer
        self.verbose = verbose

        self.early_stopping = False
        self.patience, self.min_epochs = None, None
        self.best_epoch, self.best_value = None, None
        self._early_stopping_check = None

        self._optuna_callback = None

    def set_early_stopping(self, patience, min_epochs=0, objective="loss"):
        self.early_stopping = True
        self.patience = patience
        self.min_epochs = min_epochs

        self.best_epoch = 1
        self.best_value = np.inf if objective == "loss" else -np.inf

        if objective == "loss":
            def check(val_loss, val_acc):
                if val_loss < self.best_value:
                    self.best_epoch = self.epoch
                    self.best_value = val_loss
        else:
            def check(val_loss, val_acc):
                if val_acc > self.best_value:
                    self.best_epoch = self.epoch
                    self.best_value = val_acc

        self._early_stopping_check = check

    def set_optuna_trial_prunning(self, trial, objective="loss"):
        def optuna_callback(val_loss, val_acc):
            trial.report(val_loss if objective == "loss" else val_acc, self.epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        self._optuna_callback = optuna_callback

    def get_best(self, key):
        return self.history[key][self.best_epoch - 1]

    def train(self, train_loader, val_loader, epochs):
        while self.epoch < epochs:
            self.epoch += 1
            train_loss, train_acc, val_loss, val_acc = self.train_epoch(train_loader, val_loader)
            if self.verbose:
                print(f"Epoch {self.epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

            if self.writer:
                self._write_to_history(train_loss, train_acc, val_loss, val_acc)

            if self.early_stopping:
                self._early_stopping_check(val_loss, val_acc)
                if self.epoch - self.best_epoch > self.patience:
                    break

            if self._optuna_callback:
                self._optuna_callback(val_loss, val_acc)

    def _write_to_history(self, train_loss, train_acc, val_loss, val_acc):
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)

        if self.writer:
            self.writer.add_scalar('Loss/train', train_loss, self.epoch)
            self.writer.add_scalar('Loss/test', val_loss, self.epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, self.epoch)
            self.writer.add_scalar('Accuracy/test', val_acc, self.epoch)

    def train_epoch(self, train_loader, val_loader):
        self.model.train()

        train_loss = 0
        train_correct = 0
        for batch in train_loader:
            batch = batch.to(self.device)
            out = self.model(batch.x, batch.edge_index, batch.batch)
            loss = self.criterion(out, batch.y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_correct += out.argmax(dim=1).eq(batch.y).sum().item()

        self.model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.batch)
                val_loss += self.criterion(out, batch.y).item()
                val_correct += out.argmax(dim=1).eq(batch.y).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / len(train_loader.dataset)
        val_loss /= len(val_loader)
        val_acc = val_correct / len(val_loader.dataset)

        return train_loss, train_acc, val_loss, val_acc
