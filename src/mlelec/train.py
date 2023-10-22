import argparse
import torch
import sys
import os
import warnings

from mlelec.train_setup import Trainer
from mlelec.models import get_model
from mlelec.datasets.dataset_utils import get_dataset

from mlelec.data.dataset import precomputer_molecules

all_molecules = [mol.name.lower() for mol in precomputer_molecules]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mol",
    type=str,
    default="water",
    help="u, choose from (case insensitive)",
)
parser.add_argument("--basis", type=str, default="sto-3g", help="basis set")
parser.add_argument(
    "--data_folder",
    type=str,
    default="./data",
    help="directory root to save simulation data",
)
parser.add_argument(
    "--results_folder",
    type=str,
    default="./results",
    help="directory root to save model checkpoints and samples",
)

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
    "--scale_data",
    type=eval,
    default=True,
    help="set True to scale data points by dividing by the dataset's std. This should be disabled when training with energy priors, because otherwise the bond distances are perturbed.",
)
parser.add_argument(
    "--pick_checkpoint",
    type=str,
    default="best",
    help="last to evaluate on the last saved model. Best to evaluate on the best crossvalidated model (which can be noisy sometimes)",
)
parser.add_argument(
    "--start_from_last_saved",
    type=eval,
    default=False,
    help="Load last saved checkpoint and start from there...",
)

parser.add_argument(
    "--backbone_network",
    type=str,
    default="gnn",
    help="gnn, cgnet, graph-transformer",
)
parser.add_argument(
    "--save_all_checkpoints",
    type=eval,
    default=False,
    help="set True to do save all checkpoints not only the best crossvalidated one",
)


args = parser.parse_args()

# change some arguments depending on the situation

if __name__ == "__main__":
    trainset, valset, testset = get_dataset(
        args.mol,
        args.data_folder,
        traindata_subset=args.traindata_subset,
        shuffle_before_splitting=args.shuffle_data_before_splitting,
    )

    norm_factor = trainset.std if args.scale_data else 1.0

    # Set device
    # Note: Code does not work for cpu in current form
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Backbone
    model = get_model(args, trainset, energy_prior, device)
    print(model)

    # Overall
    DDPM_model = GaussianDiffusion(
        model=model,
        features=trainset.bead_onehot,
        num_atoms=trainset.num_beads,
        timesteps=args.diffusion_steps,
        norm_factor=norm_factor,
        loss_weights=args.loss_weights,
        objective=args.ddpm_objective,
        forces_reg_weight=args.forces_reg_weight,
    )

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
        ema_decay=args.ema_decay,
        save_and_sample_every=args.eval_interval,
        num_saved_samples=args.num_samples,
        topology=trainset.topology,
        results_folder=args.results_folder,
        data_aug=args.data_aug,
        tb_folder=args.tensorboard_folder,
        experiment_name=args.experiment_name,
        weight_decay=args.weight_decay,
        log_tensorboard_interval=args.log_tensorboard_interval,
        num_samples_final_eval=args.num_samples_final_eval,
        min_lr_cosine_anneal=args.min_lr_cosine_anneal,
        eval_langevin=args.eval_langevin,
        langevin_timesteps=args.langevin_timesteps,
        langevin_stepsize=args.langevin_stepsize,
        langevin_t_diffs=args.langevin_t_diff,
        start_from_last_saved=args.start_from_last_saved,
        pick_checkpoint=args.pick_checkpoint,
        iterations_on_val=args.iterations_on_val,
        t_diff_interval=args.t_diff_interval,
        parallel_tempering=args.parallel_tempering,
        save_all_checkpoints=args.save_all_checkpoints,
    )

    # Training
    trainer.train()
