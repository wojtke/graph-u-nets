import argparse

import torch
from torch_geometric.loader import DataLoader

from callbacks import EarlyStopping, TensorBoardWriter
from dataset_utils import get_out_channels, get_criterion, load_ogb
from hyperparams import HyperparamsSpace
from model_selection import select_hyperparams
from models import GNN
from training import Trainer
from utils import set_reproducibility, print_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv", help="dataset")
    parser.add_argument("--pool", type=str, default="topk", help="Pooling method")
    parser.add_argument("--conv", type=str, default="gcn", help="Convolution method")
    parser.add_argument("--selection-trials", type=int, default=1,
                        help="Number of trials for hyperparameter selection")
    parser.add_argument("--hyperparams-space", type=str, default="hyperparams_space.yml")
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    study_name = f"{args.dataset}/OGB_{args.conv}_{args.pool}_ok"

    set_reproducibility(args.seed)

    # Load the dataset related things
    dataset = load_ogb(args.dataset)

    split_idx = dataset.get_idx_split()

    print(f"Dataset loaded: {args.dataset} - Task: {dataset.task_type} - Metric: {dataset.eval_metric}")
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
        n_trials=args.selection_trials,
        pruning=False
    )
    print(f"Selected hyperparams: \n{hyperparams}")

    train_loader = DataLoader(dataset[split_idx["train"]], hyperparams.batch_size, shuffle=True)
    val_loader = DataLoader(dataset[split_idx["valid"]], hyperparams.batch_size, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], hyperparams.batch_size, shuffle=False)

    # Initialize the model and the trainer using selected hyperparams
    model = GNN(
        in_channels=dataset.num_features,
        out_channels=get_out_channels(dataset),
        hyperparams=hyperparams
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.lr, weight_decay=hyperparams.weight_decay)
    criterion = get_criterion(dataset)

    trainer = Trainer(model, optimizer, criterion, dataset.eval_metric)
    trainer.add_callbacks(
        EarlyStopping(patience=hyperparams.patience, save_path=f"runs/{study_name}/model.pt"),
        #TensorBoardWriter(log_dir=f"runs"),
    )

    trainer.train(train_loader, val_loader, epochs=hyperparams.max_epochs)

    # Evaluate the model on the test set.
    model.load_state_dict(torch.load(f"runs/{study_name}/model.pt"))

    result = trainer.evaluate(test_loader, model)

    print(f"Test results: {result}")

    with open(f"runs/{study_name}/results", "w") as f:
        f.write(f"Result: {result}\n")
