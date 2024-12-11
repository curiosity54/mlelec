import matplotlib.pyplot as plt


def plot_losses(history):
    """
    Plots training and validation losses for total, eva, polar, and dipole on separate plots.

    Args:
        history: A list of dictionaries containing loss metrics for each epoch.
    """

    train_total_loss = [epoch["train_loss"] for epoch in history]
    train_loss_eva = [epoch["train_loss_eva"] for epoch in history]
    train_loss_polar = [epoch["train_loss_polar"] for epoch in history]
    train_loss_dipole = [epoch["train_loss_dipole"] for epoch in history]

    val_total_loss = [epoch["val_loss"] for epoch in history]
    val_loss_eva = [epoch["val_loss_eva"] for epoch in history]
    val_loss_polar = [epoch["val_loss_polar"] for epoch in history]
    val_loss_dipole = [epoch["val_loss_dipole"] for epoch in history]

    # Create the figure
    plt.figure(figsize=(14, 6))

    # Training Losses
    plt.subplot(1, 2, 1)
    plt.loglog(train_total_loss, label="Total Loss")
    plt.loglog(train_loss_eva, "--", label="EVA Loss")
    plt.loglog(train_loss_polar, "--", label="Polar Loss")
    plt.loglog(train_loss_dipole, "--", label="Dipole Loss")
    plt.title("Training Losses", fontsize=15)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(True)

    # Validation Losses
    plt.subplot(1, 2, 2)
    plt.loglog(val_total_loss, label="Total Loss")
    plt.loglog(val_loss_eva, "--", label="EVA Loss")
    plt.loglog(val_loss_polar, "--", label="Polar Loss")
    plt.loglog(val_loss_dipole, "--", label="Dipole Loss")
    plt.title("Validation Losses", fontsize=15)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(True)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()
