import argparse
import torch 
import hickle 
import sys
import os
import warnings

from mlelec.train_setup import ModelTrainer
from mlelec.data.dataset import MLDataset, MoleculeDataset, precomputed_molecules, get_dataloader
from mlelec.features.acdc import compute_features_for_target
from mlelec.models.linear import LinearTargetModel
parser = argparse.ArgumentParser()
# parser.add_argument("--device", type=str, default='cuda')
#----data related-----
parser.add_argument(
    "--molecule",
    type=str,
    default="water",
    help="name of molecule/dataset to train ",
)
parser.add_argument("--frame_begin", type=int, default=0)
parser.add_argument("--frame_end", type=int, default=-1)

parser.add_argument("--use_precomputed_data", type=bool, default=True)
parser.add_argument("--basis", type=str, default="sto-3g", help="basis set")
parser.add_argument(
    "--data_folder",
    type=str,
    default="./data",
    help="path to data",
)
parser.add_argument("--target", type=str, nargs='+', default="fock")
parser.add_argument("--aux_data", type=str, nargs='+')
#----- model related-------
parser.add_argument(
    "--model_type",
    type=str,
    default="acdc",
    help="acdc, se3-transformer",
)
parser.add_argument("--model_strategy", type=str, default="coupled", help="coupled, uncoupled")

# ---- ACDC related -------
parser.add_argument("--feature_path", type=str, default=None)
parser.add_argument("--nlayers", type=int, default=1)
parser.add_argument("--nhidden", type=int, default=16)
parser.add_argument("--nmax", type=int, default=6)
parser.add_argument("--lmax", type=int, default=4)
parser.add_argument("--cutoff", type=float, default=4.0)
#---- training -------
parser.add_argument("--train_fraction", type=float, default=0.7)
parser.add_argument("--val_fraction", type=float, default=0.1)
parser.add_argument("--test_fraction", type=float, default=0.2)
parser.add_argument(
    "--batch_size",
    type=int,
    default=256,
    help="batch sized used in trianing and validation",
)
parser.add_argument(
    "--learning_rate", type=float, default=2e-4, help="learning rate for Adam"
)
parser.add_argument("--nonlinearity", type=str, default=None, help="SiLU")
parser.add_argument("--optimizer", type=str, default="Adam")
parser.add_argument(
    "--max_epochs",
    type=int,
    default=2500,
    help="Max number of iterations to train the model",
)
#----checkpoint and logging -----
parser.add_argument(
    "--save_path",
    type=str,
    default="./logdir",
    help=" path to save model checkpoints and predictions",
)
parser.add_argument(
    "--log_interval",
    type=int,
    default=1000,
    help="interval at which to calculate and log evaluation metrics",
)
parser.add_argument(
    "--start_from_checkpoint",
    type=bool,
    default=False,
    help="Whether to load saved checkpoint and start from there, specify checkpoint by --starting_checkpoint",
)
parser.add_argument("--starting_checkpoint", 
                    type=str, default='last', help='last, best, or checkpoint number to begin')
parser.add_argument(
    "--save_all_checkpoints",
    type=eval,
    default=False,
    help="set True to do save all checkpoints not only the best crossvalidated one",
)
#---- end of arguments ----
args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
precomp_datasets = [mol.name.lower() for mol in precomputed_molecules]
frame_slice = slice(args.frame_begin, args.frame_end)
mol_dataset =  MoleculeDataset(mol_name= args.molecule, 
                               frame_slice=args.frame_slice, 
                               aux = args.aux_data, 
                               target=args.target,
                               use_precomputed=args.use_precomputed_data)

ml_data = MLDataset(
    molecule_data=mol_dataset,
    model_strategy=args.model_strategy,
    device= args.device,
    shuffle=True,
    shuffle_seed=5380,
    train_frac = args.train_frac, 
    val_frac = args.val_frac
) 

# instantiate model based on args['model_type'] -'linear', 'se3-transformer'..
def instantiate_model(args, dataset: MLDataset, device):
    if args.model_type == "acdc":
        # check if features are provided
        model = LinearTargetModel(dataset, features=features, device=device)

    elif args.model_type == "se3-transformer":
        raise NotImplementedError
        # model = SE3TransformerTargetModel(dataset, device)
    else:
        raise NotImplementedError
    return model


if __name__ == "__main__":

    if args.model_type == "acdc":
    # try to load saved features by default but
    # check if hypers provided to generate features if not
    # else compute default features
        if args.feature_path is not None:
            features = hickle.load(args.feature_path)
        else:
            acdc_hypers = {"cutoff":args.cutoff,
                        "max_radial": args.nmax,
                        "max_angular": args.lmax, 
                        }

            features=compute_features_for_target(dataset=ml_data, hypers=acdc_hypers, device=args.device)

    train_dl, val_dl, test_dl = get_dataloader(ml_data, model_return="tensor")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = instantiate_model(args, ml_data, device)
    print(model)

    # Trainer
    trainer = ModelTrainer(
        model=model.to(device),
        dataset_split=(train_dl, val_dl, test_dl),
        mol_name=args.mol,
        train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        log_interval=args.eval_interval,
        save_path=args.save_path,
        start_from_checkpoint=args.start_from_checkpoint,
        checkpoint=args.starting_checkpoint,
        save_all_checkpoints=args.save_all_checkpoints,
        device = args.device
    )

    # Training
    trainer.train()
