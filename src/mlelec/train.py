import argparse
import torch
import sys
import os
import warnings

from mlelec.train_setup import Trainer
from mlelec.models import get_model
from mlelec.datasets.dataset_utils import get_dataset
from mlelec.features.acdc import compute_features_for_target

from mlelec.data.dataset import precomputed_molecules

all_ds = [mol.name.lower() for mol in precomputed_molecules]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="water",
    help="name of molecule/dataset to train ",
)
parser.add_argument("--basis", type=str, default="sto-3g", help="basis set")
parser.add_argument(
    "--data_folder",
    type=str,
    default="./data",
    help="path to data",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="./logdir",
    help=" path to save model checkpoints and predictions",
)

parser.add_argument("--feature_path", type=str, default=None)
parser.add_argument("--hypers", type=dict, default=None)


parser.add_argument(
    "--batch_size",
    type=int,
    default=256,
    help="batch sized used in trianing and validation",
)
parser.add_argument(
    "--learning_rate", type=float, default=2e-4, help="learning rate for Adam"
)

parser.add_argument(
    "--max_epochs",
    type=int,
    default=25000,
    help="Max number of iterations to train the model",
)

parser.add_argument(
    "--log_interval",
    type=int,
    default=1000,
    help="interval at which to calculate and log evaluation metrics",
)


parser.add_argument(
    "--from_last_check",
    type=bool,
    default=False,
    help="Load last saved checkpoint and start from there...",
)

parser.add_argument(
    "--model_type",
    type=str,
    default="linear",
    help="linear, kernel, nonlinear, se3-transformer",
)
parser.add_argument(
    "--save_checkpoints",
    type=eval,
    default=False,
    help="set True to do save all checkpoints not only the best crossvalidated one",
)


args = parser.parse_args()

if args.model_type == "acdc" and args.feature_path is None:
    # try to load saved features by default but
    # check if hypers provided to generate features if not
    # else compute default features
    if args.hypers is None:
        warnings.warn("No hypers provided. Computing default features")
        hypers = {
            "n_layers": 3,
            "n_hidden": 128,
            "activation": "tanh",
            "norm": True,
            "bias": False,
        }

    compute_features_for_target(dataset=dataset, hypers=hypers, device=device)


# instantiate model based on args['model_type'] -'linear', 'se3-transformer'..
def instantiate_model(args, dataset: MLDataset, device):
    if args.model_type == "linear":
        # check if features are provided
        model = LinearTargetModel(dataset, features=features, device=device)

    elif args.model_type == "se3-transformer":
        raise NotImplementedError
        # model = SE3TransformerTargetModel(dataset, device)
    else:
        raise NotImplementedError
    return model


# change some arguments depending on the situation


if __name__ == "__main__":
    trainset, valset, testset = get_dataset(
        args.mol,
        args.data_folder,
        traindata_subset=args.traindata_subset,
        shuffle_before_splitting=args.shuffle_data_before_splitting,
    )

    norm_factor = trainset.std if args.scale_data else 1.0

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Backbone
    model = get_model(args, trainset, metrics, device)
    print(model)

    # Overall
    # DDPM_model = GaussianDiffusion(
    #     model=model,
    #     features=trainset.bead_onehot,
    #     num_atoms=trainset.num_beads,
    #     timesteps=args.diffusion_steps,
    #     norm_factor=norm_factor,
    #     loss_weights=args.loss_weights,
    #     objective=args.ddpm_objective,
    #     forces_reg_weight=args.forces_reg_weight,
    # )

    # Trainer
    trainer = Trainer(
        DDPM_model.to(device),
        (trainset, valset, testset),
        args.mol,
        args,
        train_batch_size=args.batch_size,
        train_lr=args.learning_rate,
        train_num_steps=args.train_iter,
        gradient_accumulate_every=1,
        save_and_sample_every=args.eval_interval,
        results_folder=args.results_folder,
        experiment_name=args.experiment_name,
        start_from_last_saved=args.start_from_last_saved,
        pick_checkpoint=args.pick_checkpoint,
        save_all_checkpoints=args.save_all_checkpoints,
    )

    # Training
    trainer.train()
