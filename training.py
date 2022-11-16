import torch


def train_epoch(model, optimizer, criterion, train_loader, val_loader, device):
    model.train()

    for batch in train_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss += criterion(out, batch.y)
            correct += (out.argmax(dim=1) == batch.y).sum()

    loss = loss / len(val_loader.dataset)
    acc = correct / len(val_loader.dataset)

    return loss.cpu().item(), acc.cpu().item()


class EarlyStopping:
    def __init__(self, patience, objective="minimize", min_epochs=0, verbose=False):
        self.objective = objective
        self.patience = patience
        self.min_epochs = min_epochs
        self.verbose = verbose

        self.best_objective = None
        self.best_epoch = None
        self.epoch = 0

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return type == EarlyStoppingException

    def check(self, objective):
        objective = objective.item() if torch.is_tensor(objective) else objective
        self.epoch += 1
        if self.best_objective is None:
            self.best_objective = objective
            self.best_epoch = self.epoch

        if (
            self.objective == "minimize"
            and objective < self.best_objective
            or self.objective == "maximize"
            and objective > self.best_objective
        ):
            self.best_objective = objective
            self.best_epoch = self.epoch

        if (
            self.epoch - self.best_epoch > self.patience
            and self.epoch > self.min_epochs
        ):
            if self.verbose:
                print(
                    f"Early stopping after {self.epoch - 1} epochs with best "
                    f"objective of {self.best_objective:.4f} at epoch {self.best_epoch}."
                )
            raise EarlyStoppingException()


class EarlyStoppingException(Exception):
    """Exception to stop early."""

    pass
