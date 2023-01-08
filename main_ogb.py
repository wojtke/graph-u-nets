import argparse

import torch
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from hyperparams import HyperparamsSpace
from mappings import get_evaluation_metric
from model_selection import select_hyperparams
from models import GNN
from training import Trainer
from utils import set_reproducibility, print_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv", help="dataset")
    parser.add_argument("--pool", type=str, default="topk", help="Pooling method")
    parser.add_argument("--conv", type=str, default="gcn", help="Convolution method")
    parser.add_argument("--selection-trials", type=int, default=50,
                        help="Number of trials for hyperparameter selection")
    parser.add_argument("--hyperparams-space", type=str, default="hyperparams_space.yml")
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help="Device to be used for training")
    return parser.parse_args()


@torch.no_grad()
def eval(model, loader, evaluator, device):
    model.eval()
    y_true = []
    y_pred = []

    for batch in loader:
        batch = batch.to(device)
        batch.x = batch.x.float()

        pred = model(batch.x, batch.edge_index, batch.batch)
        y_true.append(batch.y.detach().cpu())
        y_pred.append(pred.detach().cpu())

    input_dict = {
        "y_true": torch.cat(y_true, dim=0).numpy(),
        "y_pred": torch.cat(y_pred, dim=0).numpy()[..., 0].reshape(-1, 1),
    }

    return evaluator.eval(input_dict)


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    study_name = f"{args.dataset}/OGB_{args.conv}_{args.pool}"

    set_reproducibility(args.seed)
    device = torch.device(int(args.device)) if args.device.isdigit() else torch.device(args.device)

    # Load the dataset related things
    dataset = PygGraphPropPredDataset(name=args.dataset)
    task, evaluation_metric = dataset.task_type, dataset.eval_metric
    evaluation_metric = get_evaluation_metric(evaluation_metric)

    split_idx = dataset.get_idx_split()

    print(f"Dataset loaded: {args.dataset} - Task: {task} - Metric: {evaluation_metric}")
    print(f"Loaded splits. Train: {len(split_idx['train'])} - "
          f"Val: {len(split_idx['valid'])} - "
          f"Test: {len(split_idx['test'])}")

    hyperparams_space = HyperparamsSpace.load(
        path=args.hyperparams_space,
        conv=args.conv,
        pool=args.pool,
    )
    print(f"Hyperparams space: \n{hyperparams_space}")

    # Select the best hyperparameters using optuna.

    hyperparams = select_hyperparams(
        dataset,
        split=(split_idx["train"], split_idx["valid"]),
        study_name=f"{study_name}",
        hyperparams_space=hyperparams_space,
        metric=evaluation_metric,
        n_trials=args.selection_trials,
        task=task,
        pruning=False,
        device=device,
    )
    print(f"Selected hyperparams: \n{hyperparams}")

    train_loader = DataLoader(dataset[split_idx["train"]], hyperparams.batch_size, shuffle=True)
    val_loader = DataLoader(dataset[split_idx["valid"]], hyperparams.batch_size, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], hyperparams.batch_size, shuffle=False)

    # Initialize the model and the trainer using selected hyperparams
    model = GNN(in_channels=dataset.num_features, out_channels=dataset.num_classes, hyperparams=hyperparams)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams.lr,
        weight_decay=hyperparams.weight_decay,
    )
    criterion = torch.nn.MSELoss() if task == "regression" else torch.nn.CrossEntropyLoss()

    # Train the model. Use early stopping on the validation set.
    writer = SummaryWriter(f"runs/{study_name}/assessment")
    trainer = Trainer(model, optimizer, criterion, metric=evaluation_metric, device=device, writer=writer, verbose=True)
    trainer.set_early_stopping(
        patience=hyperparams.patience,
        min_epochs=hyperparams.min_epochs,
        save_path=f"runs/{study_name}/model.pt",
    )
    trainer.train(train_loader, val_loader, epochs=hyperparams.max_epochs)

    # Evaluate the model on the test set.
    evaluator = Evaluator(args.dataset)
    model.load_state_dict(torch.load(f"runs/{study_name}/model.pt"))

    result = eval(
        model=model,
        loader=test_loader,
        evaluator=Evaluator(args.dataset),
        device=device,
    )

    print(f"Test results: {result}")

    with open(f"runs/{study_name}/result", "w") as f:
        f.write(
            f"Result: {result}\n"
        )
