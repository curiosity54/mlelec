import os
import time

import ase
import matplotlib.pyplot as plt
import metatensor
import numpy as np
import scipy
import torch
from ase.units import Hartree
from IPython.utils import io
from metatensor import Labels
from tqdm import tqdm

import mlelec.metrics as mlmetrics
from mlelec.data.dataset import MoleculeDataset, get_dataloader
from mlelec.features.acdc import compute_features_for_target
from src.mlelec.data.dataset import MLDataset
from src.mlelec.models.linear import LinearTargetModel
from src.mlelec.utils.dipole_utils import compute_batch_polarisability, instantiate_mf


torch.set_default_dtype(torch.float64)

# ------------------ CHANGE THE PARAMETERS -------------
NUM_FRAMES = 1000
BATCH_SIZE = 100
NUM_EPOCHS = 1000
SHUFFLE_SEED = 1234
TRAIN_FRAC = 0.7
TEST_FRAC = 0.1
VALIDATION_FRAC = 0.2

LR = 1e-4
VAL_INTERVAL = 1
W_EVA = 1e4
W_DIP = 1e3
W_POL = 1e2
DEVICE = "cpu"

ORTHOGONAL = False  # set to 'FALSE' if working in the non-orthogonal basis
FOLDER_NAME = "multitask_learn_adam_with_noise"
NOISE = True
# ---------------------------------------------------------

os.makedirs(FOLDER_NAME, exist_ok=True)
os.makedirs(f"{FOLDER_NAME}/model_output", exist_ok=True)


def save_parameters(file_path, **params):
    with open(file_path, "w") as file:
        for key, value in params.items():
            file.write(f"{key}: {value}\n")


# Call the function with your parameters
save_parameters(
    f"{FOLDER_NAME}/parameters.txt",
    NUM_FRAMES=NUM_FRAMES,
    BATCH_SIZE=BATCH_SIZE,
    NUM_EPOCHS=NUM_EPOCHS,
    SHUFFLE_SEED=SHUFFLE_SEED,
    TRAIN_FRAC=TRAIN_FRAC,
    TEST_FRAC=TEST_FRAC,
    VALIDATION_FRAC=VALIDATION_FRAC,
    LR=LR,
    VAL_INTERVAL=VAL_INTERVAL,
    W_EVA=W_EVA,
    W_DIP=W_DIP,
    W_POL=W_POL,
    DEVICE=DEVICE,
    ORTHOGONAL=ORTHOGONAL,
    FOLDER_NAME=FOLDER_NAME,
    NOISE=NOISE,
)


def drop_zero_blocks(train_tensor, val_tensor, test_tensor):
    for i1, b1 in train_tensor.items():
        if b1.values.shape[0] == 0:
            train_tensor = metatensor.drop_blocks(
                train_tensor, Labels(i1.names, i1.values.reshape(1, -1))
            )

    for i2, b2 in val_tensor.items():
        if b2.values.shape[0] == 0:
            val_tensor = metatensor.drop_blocks(
                val_tensor, Labels(i2.names, i2.values.reshape(1, -1))
            )

    for i3, b3 in test_tensor.items():
        if b3.values.shape[0] == 0:
            test_tensor = metatensor.drop_blocks(
                test_tensor, Labels(i3.names, i3.values.reshape(1, -1))
            )

    return train_tensor, val_tensor, test_tensor


# loss function to have different combinations of losses
def loss_fn_combined(
    ml_data,
    pred_focks,
    orthogonal,
    mfs,
    indices,
    loss_fn,
    frames,
    eigval,
    polar,
    dipole,
    var_polar,
    var_dipole,
    var_eigval,
    weight_eigval=1.0,
    weight_polar=1.0,
    weight_dipole=1.0,
):

    pred_dipole, pred_polar, pred_eigval = compute_batch_polarisability(
        ml_data, pred_focks, indices, mfs, orthogonal
    )

    loss_polar = loss_fn(frames, pred_polar, polar) / var_polar
    loss_dipole = loss_fn(frames, pred_dipole, dipole) / var_dipole
    loss_eigval = loss_fn(frames, pred_eigval, eigval) / var_eigval

    # weighted sum of the various loss contributions
    return (
        weight_eigval * loss_eigval
        + weight_dipole * loss_dipole
        + weight_polar * loss_polar,
        loss_eigval,
        loss_dipole,
        loss_polar,
    )


start_time_pred_lbfgs = time.time()

molecule_data = MoleculeDataset(
    mol_name="qm7",
    use_precomputed=True,
    path="examples/data/qm7",
    aux_path="examples/data/qm7/sto-3g",
    frame_slice=slice(0, NUM_FRAMES),
    device=DEVICE,
    aux=["overlap", "orbitals"],
    lb_aux=["overlap", "orbitals"],
    target=["fock", "dipole_moment", "polarisability"],
    lb_target=["fock", "dipole_moment", "polarisability"],
)

ml_data = MLDataset(
    molecule_data=molecule_data,
    device=DEVICE,
    model_strategy="coupled",
    shuffle=True,
    shuffle_seed=SHUFFLE_SEED,
    orthogonal=ORTHOGONAL,
)

ml_data._split_indices(
    train_frac=TRAIN_FRAC, val_frac=VALIDATION_FRAC, test_frac=TEST_FRAC
)

hypers = {
    "cutoff": 3.0,
    "max_radial": 6,
    "max_angular": 4,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 1,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.1}},
}
hypers_pair = {
    "cutoff": 4.0,
    "max_radial": 6,
    "max_angular": 4,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 1,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.1}},
}

ml_data._set_features(
    compute_features_for_target(
        ml_data, device=DEVICE, hypers=hypers, hypers_pair=hypers_pair
    )
)

train_dl, val_dl, test_dl = get_dataloader(
    ml_data, model_return="blocks", batch_size=BATCH_SIZE
)


ml_data.target_train, ml_data.target_val, ml_data.target_test = drop_zero_blocks(
    ml_data.target_train, ml_data.target_val, ml_data.target_test
)

ml_data.feat_train, ml_data.feat_val, ml_data.feat_test = drop_zero_blocks(
    ml_data.feat_train, ml_data.feat_val, ml_data.feat_test
)

model = LinearTargetModel(
    dataset=ml_data, nlayers=1, nhidden=16, bias=False, device=DEVICE
)

pred_ridges, ridges = model.fit_ridge_analytical(
    alpha=np.logspace(-8, 3, 12),
    cv=3,
    set_bias=False,
)

pred_fock = model.forward(
    ml_data.feat_train,
    return_type="tensor",
    batch_indices=ml_data.train_idx,
    ridge_fit=True,
    add_noise=NOISE,
)

ref_polar_lb = molecule_data.lb_target["polarisability"]
ref_dip_lb = molecule_data.lb_target["dipole_moment"]

ref_eva_lb = []
for i in range(len(molecule_data.lb_target["fock"])):
    f = molecule_data.lb_target["fock"][i]
    s = molecule_data.lb_aux_data["overlap"][i]
    eig = scipy.linalg.eigvalsh(f, s)
    ref_eva_lb.append(torch.from_numpy(eig))

ref_polar = molecule_data.target["polarisability"]
ref_dip = molecule_data.target["dipole_moment"]
ref_eva = []
for i in range(len(molecule_data.target["fock"])):
    f = molecule_data.target["fock"][i]
    s = molecule_data.aux_data["overlap"][i]
    eig = scipy.linalg.eigvalsh(f, s)
    ref_eva.append(torch.from_numpy(eig))

var_eigval = torch.cat([ref_eva_lb[i].flatten() for i in range(len(ref_eva_lb))]).var()
var_dipole = torch.cat([ref_dip_lb[i].flatten() for i in range(len(ref_dip_lb))]).var()
var_polar = torch.cat(
    [ref_polar_lb[i].flatten() for i in range(len(ref_polar_lb))]
).var()

loss_fn = getattr(mlmetrics, "mse_qm7")

with io.capture_output() as captured:
    all_mfs, fockvars = instantiate_mf(
        ml_data,
        fock_predictions=None,
        batch_indices=list(range(len(ml_data.structures))),
    )

best = float("inf")
early_stop_criteria = 10
loss_fn = getattr(mlmetrics, "mse_qm7")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.5,
    patience=20,
)

val_interval = VAL_INTERVAL
early_stop_count = 0

# Initialize lists to store different losses
losses = []
val_losses = []
losses_eva = []
val_losses_eva = []
losses_polar = []
val_losses_polar = []
losses_dipole = []
val_losses_dipole = []

iterator = tqdm(range(NUM_EPOCHS))
for epoch in range(NUM_EPOCHS):
    model.train(True)
    train_loss = 0
    train_loss_eva = 0
    train_loss_polar = 0
    train_loss_dipole = 0

    for data in train_dl:
        optimizer.zero_grad()
        idx = data["idx"]

        # Forward pass
        pred = model(
            data["input"], return_type="tensor", batch_indices=[i.item() for i in idx]
        )
        train_polar_ref = ref_polar_lb[[i.item() for i in idx]]
        train_dip_ref = ref_dip_lb[[i.item() for i in idx]]
        train_eva_ref = [
            ref_eva_lb[i][: ml_data.target.tensor[i].shape[0]] for i in idx
        ]

        loss, loss_eva, loss_dipole, loss_polar = loss_fn_combined(
            ml_data,
            pred,
            ORTHOGONAL,
            all_mfs,
            idx,
            loss_fn,
            data["frames"],
            train_eva_ref,
            train_polar_ref,
            train_dip_ref,
            var_polar,
            var_dipole,
            var_eigval,
            W_EVA,
            W_POL,
            W_DIP,
        )
        train_loss += loss.item()
        train_loss_eva += loss_eva.item()
        train_loss_polar += loss_polar.item()
        train_loss_dipole += loss_dipole.item()

        # Backward pass
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

    avg_train_loss = train_loss / len(train_dl)
    avg_train_loss_eva = train_loss_eva / len(train_dl)
    avg_train_loss_polar = train_loss_polar / len(train_dl)
    avg_train_loss_dipole = train_loss_dipole / len(train_dl)

    losses.append(avg_train_loss)
    losses_eva.append(avg_train_loss_eva)
    losses_polar.append(avg_train_loss_polar)
    losses_dipole.append(avg_train_loss_dipole)

    lr = optimizer.param_groups[0]["lr"]

    model.eval()
    if epoch % val_interval == 0:
        val_loss = 0
        val_loss_eva = 0
        val_loss_polar = 0
        val_loss_dipole = 0

        for data in val_dl:
            idx = data["idx"]
            val_pred = model(
                data["input"],
                return_type="tensor",
                batch_indices=[i.item() for i in idx],
            )

            val_polar_ref = ref_polar_lb[[i.item() for i in idx]]
            val_dip_ref = ref_dip_lb[[i.item() for i in idx]]
            val_eva_ref = [
                ref_eva_lb[i][: ml_data.target.tensor[i].shape[0]] for i in idx
            ]

            vloss, vloss_eva, vloss_dipole, vloss_polar = loss_fn_combined(
                ml_data,
                val_pred,
                ORTHOGONAL,
                all_mfs,
                idx,
                loss_fn,
                data["frames"],
                val_eva_ref,
                val_polar_ref,
                val_dip_ref,
                var_polar,
                var_dipole,
                var_eigval,
                W_EVA,
                W_POL,
                W_DIP,
            )

            val_loss += vloss.item()
            val_loss_eva += vloss_eva.item()
            val_loss_polar += vloss_polar.item()
            val_loss_dipole += vloss_dipole.item()

        avg_val_loss = val_loss / len(val_dl)
        avg_val_loss_eva = val_loss_eva / len(val_dl)
        avg_val_loss_polar = val_loss_polar / len(val_dl)
        avg_val_loss_dipole = val_loss_dipole / len(val_dl)

        val_losses.append(avg_val_loss)
        val_losses_eva.append(avg_val_loss_eva)
        val_losses_polar.append(avg_val_loss_polar)
        val_losses_dipole.append(avg_val_loss_dipole)

        new_best = avg_val_loss < best
        if new_best:
            best = val_loss
            # torch.save(model.state_dict(), 'model_output_combined/best_model_dipole.pt')
            early_stop_count = 0
        else:
            early_stop_count += 1
        if early_stop_count > early_stop_criteria:
            print(f"Early stopping at epoch {epoch}")
            print(
                f"Epoch {epoch}, train loss {avg_train_loss}, val loss {avg_val_loss}"
            )
            # Save last best model
            break

        scheduler.step(avg_val_loss)

    if epoch % 50 == 0:
        torch.save(
            model.state_dict(), f"{FOLDER_NAME}/model_output/model_epoch_{epoch}.pt"
        )

    if epoch % 1 == 0:
        print(
            "Epoch:",
            epoch,
            "train loss:",
            f"{avg_train_loss:.4g}",
            "val loss:",
            f"{avg_val_loss:.4g}",
            "learning rate:",
            f"{lr:.4g}",
        )
        print(
            "Train Loss Polar:",
            f"{avg_train_loss_polar:.4g}",
            "Train Loss eva:",
            f"{avg_train_loss_eva:.4g}",
            "Train Loss dipole:",
            f"{avg_train_loss_dipole:.4g}",
        )
        print(
            "Val Loss Polar:",
            f"{avg_val_loss_polar:.4g}",
            "Val Loss eva:",
            f"{avg_val_loss_eva:.4g}",
            "Val Loss dipole:",
            f"{avg_val_loss_dipole:.4g}",
        )
    if epoch % 1 == 0:
        iterator.set_postfix(train_loss=avg_train_loss, Val_loss=avg_val_loss, lr=lr)

plt.figure()
plt.loglog(losses, label="training loss")
plt.loglog(losses_eva, "--", label="eigenvalue loss")
plt.loglog(losses_polar, "--", label="polar loss")
plt.loglog(losses_dipole, "--", label="dipole loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f"{FOLDER_NAME}/train_loss_vs_epoch.pdf", bbox_inches="tight")


plt.figure()
plt.loglog(val_losses, label="Validation loss")
plt.loglog(val_losses_eva, "--", label="eigenvalue loss")
plt.loglog(val_losses_polar, "--", label="polar loss")
plt.loglog(val_losses_dipole, "--", label="dipole loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f"{FOLDER_NAME}/val_loss_vs_epoch.pdf", bbox_inches="tight")

end_time_pred_lbfgs = time.time()
lbfgs_prediction_time = end_time_pred_lbfgs - start_time_pred_lbfgs
print(f"L-BFGS prediction time: {lbfgs_prediction_time:.4f} seconds")

with io.capture_output() as captured:
    batch_indices = ml_data.train_idx
    train_fock_predictions = model.forward(
        ml_data.feat_train, return_type="tensor", batch_indices=batch_indices
    )
    train_dipole_pred, train_polar_pred, train_eva_pred = compute_batch_polarisability(
        ml_data,
        train_fock_predictions,
        batch_indices=batch_indices,
        mfs=all_mfs,
        orthogonal=ORTHOGONAL,
    )

train_error_pol = mlmetrics.mse_qm7(
    ml_data.train_frames,
    train_polar_pred,
    ref_polar_lb[[i.item() for i in batch_indices]],
)
train_error_dip = mlmetrics.mse_qm7(
    ml_data.train_frames,
    train_dipole_pred,
    ref_dip_lb[[i.item() for i in batch_indices]],
)
train_eva_ref = [
    ref_eva_lb[i][: ml_data.target.tensor[i].shape[0]] for i in batch_indices
]
train_error_eva = mlmetrics.mse_qm7(ml_data.train_frames, train_eva_pred, train_eva_ref)

train_error_pol = mlmetrics.mse_qm7(
    ml_data.train_frames,
    train_polar_pred,
    ref_polar_lb[[i.item() for i in batch_indices]],
)
train_error_dip = mlmetrics.mse_qm7(
    ml_data.train_frames,
    train_dipole_pred,
    ref_dip_lb[[i.item() for i in batch_indices]],
)
train_eva_ref = [
    ref_eva_lb[i][: ml_data.target.tensor[i].shape[0]] for i in batch_indices
]
train_error_eva = mlmetrics.mse_qm7(ml_data.train_frames, train_eva_pred, train_eva_ref)


print(
    "Train RMSE on dipole from indirect learning {:.5f} A.U.".format(
        torch.sqrt(train_error_dip).item()
    )
)
print(
    "Train RMSE on polar from indirect learning {:.5f} A.U.".format(
        torch.sqrt(train_error_pol).item()
    )
)
print(
    "Train RMSE on MO energies from indirect learning {:.5f} eV.".format(
        torch.sqrt(train_error_eva).item() * Hartree
    )
)
with io.capture_output() as captured:
    batch_indices = ml_data.test_idx
    test_fock_predictions = model.forward(
        ml_data.feat_test,
        return_type="tensor",
        batch_indices=ml_data.test_idx,
    )

    test_dip_pred, test_polar_pred, test_eva_pred = compute_batch_polarisability(
        ml_data,
        test_fock_predictions,
        batch_indices=batch_indices,
        mfs=all_mfs,
        orthogonal=ORTHOGONAL,
    )

error_dip = mlmetrics.mse_qm7(
    ml_data.test_frames, test_dip_pred, ref_dip_lb[[i.item() for i in batch_indices]]
)
error_pol = mlmetrics.mse_qm7(
    ml_data.test_frames,
    test_polar_pred,
    ref_polar_lb[[i.item() for i in batch_indices]],
)
test_eva_ref = [
    ref_eva_lb[i][: ml_data.target.tensor[i].shape[0]] for i in batch_indices
]
error_eva = mlmetrics.mse_qm7(ml_data.test_frames, test_eva_pred, test_eva_ref)

print(
    "Test RMSE on dipole from indirect learning {:.5f} A.U.".format(
        torch.sqrt(error_dip).item()
    )
)
print(
    "Test RMSE on polar from indirect learning {:.5f} A.U.".format(
        torch.sqrt(error_pol).item()
    )
)
print(
    "Test RMSE on MO energies from indirect learning {:.5f} eV.".format(
        torch.sqrt(error_eva).item() * Hartree
    )
)
error_eva_STO3G = mlmetrics.mse_qm7(
    ml_data.test_frames, [ref_eva[i] for i in ml_data.test_idx], test_eva_ref
)
error_dip_STO3G = mlmetrics.mse_qm7(
    ml_data.test_frames, ref_dip[ml_data.test_idx], ref_dip_lb[ml_data.test_idx]
)
error_polar_STO3G = mlmetrics.mse_qm7(
    ml_data.test_frames, ref_polar[ml_data.test_idx], ref_polar_lb[ml_data.test_idx]
)

plt.figure()
for predicted, target in zip(
    test_dip_pred.detach().numpy(), ref_dip_lb[ml_data.test_idx]
):
    x = target
    y = predicted
    plt.scatter(
        x,
        y,
        color="royalblue",
        label="ML" if "ML" not in plt.gca().get_legend_handles_labels()[1] else "",
    )

# Second scatter plot
for predicted, target in zip(ref_dip[ml_data.test_idx], ref_dip_lb[ml_data.test_idx]):
    x = target
    y = predicted
    plt.scatter(
        x,
        y,
        color="chocolate",
        marker="^",
        label=(
            "STO-3G" if "STO-3G" not in plt.gca().get_legend_handles_labels()[1] else ""
        ),
    )

# Line plot
plt.plot([-2, 2], [-2, 2], linestyle="--", color="black", linewidth=1)

# Labels
plt.xlabel("Target dipoles (A.U.)")
plt.ylabel("Predicted dipoles (A.U.)")

# Text box
rmse_ml = torch.sqrt(error_dip).item()
rmse_sto3g = torch.sqrt(error_dip_STO3G).item()
plt.text(
    0.4,
    -1.8,
    f"$RMSE_{{ML}}$ = {rmse_ml:.4f} A.U.\n$RMSE_{{STO-3G}}$ = {rmse_sto3g:.4f} A.U.",
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.5),
)

# Legend
plt.legend()
plt.savefig(f"{FOLDER_NAME}/mse_dipole_indirect.pdf", bbox_inches="tight")


plt.figure()
for target, predicted in zip(test_eva_ref, test_eva_pred):
    x = target * Hartree
    y = predicted.detach().numpy() * Hartree

    x = x[x > -100]
    y = y[y > -100]
    plt.scatter(
        x,
        y,
        color="royalblue",
        label="ML" if "ML" not in plt.gca().get_legend_handles_labels()[1] else "",
    )

for target, predicted in zip(test_eva_ref, [ref_eva[i] for i in ml_data.test_idx]):
    x = target * Hartree
    y = predicted.detach().numpy() * Hartree

    x = x[x > -100]
    y = y[y > -100]
    plt.scatter(
        x,
        y,
        color="chocolate",
        marker="^",
        label=(
            "STO-3G" if "STO-3G" not in plt.gca().get_legend_handles_labels()[1] else ""
        ),
    )

plt.plot([-35, 20], [-35, 20], linestyle="--", color="black", linewidth=1)
plt.xlabel("Target MO Energies (eV)")
plt.ylabel("Predicted MO Energies (eV)")

rmse_ml = torch.sqrt(error_eva).item() * Hartree
rmse_sto3g = torch.sqrt(error_eva_STO3G).item() * Hartree
plt.text(
    5,
    -35,
    f"$RMSE_{{ML}}$ = {rmse_ml:.4f} eV\n$RMSE_{{STO-3G}}$ = {rmse_sto3g:.4f} eV.",
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.5),
)
plt.legend()
plt.savefig(f"{FOLDER_NAME}/mse_eva_indirect.pdf", bbox_inches="tight")


plt.figure()
for predicted, target in zip(
    test_polar_pred.detach().numpy(), ref_polar_lb[ml_data.test_idx]
):
    x = target
    y = predicted
    plt.scatter(
        x,
        y,
        color="royalblue",
        label="ML" if "ML" not in plt.gca().get_legend_handles_labels()[1] else "",
    )

for predicted, target in zip(
    ref_polar[ml_data.test_idx], ref_polar_lb[ml_data.test_idx]
):
    x = target
    y = predicted
    plt.scatter(
        x,
        y,
        color="chocolate",
        marker="^",
        label=(
            "STO-3G" if "STO-3G" not in plt.gca().get_legend_handles_labels()[1] else ""
        ),
    )

plt.plot([-50, 175], [-50, 175], linestyle="--", color="black", linewidth=1)
plt.xlabel("Target polarisability (A.U.)")
plt.ylabel("Predicted polarisability (A.U.)")

rmse_ml = torch.sqrt(error_pol).item()
rmse_sto3g = torch.sqrt(error_polar_STO3G).item()
plt.text(
    100,
    -50,
    f"$RMSE_{{ML}}$ = {rmse_ml:.4f} A.U.\n$RMSE_{{STO-3G}}$ = {rmse_sto3g:.4f} A.U.",
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.5),
)
plt.legend()
plt.savefig(f"{FOLDER_NAME}/mse_polar_indirect.pdf", bbox_inches="tight")
