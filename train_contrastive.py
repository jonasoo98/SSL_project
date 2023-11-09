import torch
import time
import argparse

import seaborn as sns
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataset import ContrastiveDataset
from optimizer import LARS
from loss import NT_Xent
from model import ContrastiveLearningModel

import matplotlib.pyplot as plt


def prepare_dataloader(batch_size: int = 200):
    # Fetch data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    cifar_train = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    cifar_test = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_img_array = np.array([np.array(image) for image, _ in cifar_train])
    test_img_array = np.array([np.array(image) for image, _ in cifar_test])

    # Create dataset
    train_dataset = ContrastiveDataset("train", train_img_array[:40000])
    val_dataset = ContrastiveDataset("val", train_img_array[40000:])
    test_dataset = ContrastiveDataset("test", test_img_array)

    # Create dataloader
    num_workers = 0  # means no sub-processes, needed for debugging
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_dataloader, val_dataloader, test_dataloader


def run_training(
    model,
    optimizer,
    criterion,
    train_dataloader,
    val_dataloader,
    device,
    num_epochs: int = 2,
):
    # Running training
    print(f"Starting training, {num_epochs} epochs!")
    training_losses = []
    validation_losses = []
    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        training_loss = 0
        for x_i, x_j in train_dataloader:
            optimizer.zero_grad()
            x_i, x_j = x_i.to(device), x_j.to(device)

            z_i = model(x_i)
            z_j = model(x_j)

            loss = criterion(z_i, z_j)
            loss.backward()

            optimizer.step()
            training_loss += loss.item()
            if device == torch.device("cpu"):
                print("Breaking training loop early since code is running on CPU!")
                break

        training_loss /= len(train_dataloader)
        training_losses.append(training_loss)

        model.eval()
        with torch.no_grad():
            validation_loss = 0
            for x_i, x_j in val_dataloader:
                x_i, x_j = x_i.to(device), x_j.to(device)

                z_i = model(x_i)
                z_j = model(x_j)

                loss = criterion(z_i, z_j)
                validation_loss += loss.item()
                if device == torch.device("cpu"):
                    print(
                        "Breaking validation loop early since code is running on CPU!"
                    )
                    break

            validation_loss /= len(val_dataloader)
            validation_losses.append(validation_loss)

            end = time.time()

        if epoch % 10 == 0 and epoch > 0:
            print(f"Saved model after {epoch+1} epochs!")
            model_path = f"models/encoder_{epoch+1}epochs.pth"
            torch.save(model.encoder, model_path)

            plot_results(
                training_losses=training_losses,
                validation_losses=validation_losses,
                save_path="contrastive_loss.png",
            )

        print(
            f"Epoch #{epoch+1}, "
            f"training loss: {training_loss}, "
            f"validation loss: {validation_loss}, "
            f"time: {end - start:.2f}"
        )

    print(f"Saved model after {num_epochs} epochs!")
    model_path = f"models/encoder_{num_epochs}epochs.pth"
    torch.save(model.encoder, model_path)

    return training_losses, validation_losses


def plot_results(training_losses: list, validation_losses: list, save_path: str):
    x = np.linspace(1, len(training_losses), len(training_losses))
    plt.plot(x, training_losses, label="Training")
    plt.plot(x, validation_losses, label="Validation")

    plt.xlabel("Num epochs")
    plt.ylabel("Loss")
    plt.title("Contrastive Loss")
    plt.legend()

    plt.savefig(save_path)
    plt.close()


def main():
    # Parameters
    seed = 42
    batch_size = 200
    temperature = 0.5
    learning_rate = 0.2
    weight_decay = 1e-6
    default_epochs = 2

    # Reading number of epochs from command line
    parser = argparse.ArgumentParser(description="Contrastive Learning")
    parser.add_argument(
        "--epochs", type=int, default=default_epochs, help="Number of training epochs"
    )
    args = parser.parse_args()
    num_epochs = args.epochs

    # Basic setup
    sns.set_theme()
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running {num_epochs} epochs on device: {device}!")

    train_dataloader, val_dataloader, test_dataloader = prepare_dataloader(
        batch_size=200
    )

    # Creating model, optimizer and loss function
    model = ContrastiveLearningModel().to(device)
    optimizer = LARS(
        [params for params in model.parameters() if params.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
        exclude_from_weight_decay=["batch_normalization", "bias"],
    )
    criterion = NT_Xent(batch_size=batch_size, temperature=temperature)

    training_losses, validation_losses = run_training(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        num_epochs=num_epochs,
    )

    plot_results(
        training_losses=training_losses,
        validation_losses=validation_losses,
        save_path="plots/contrastive_loss.png",
    )


if __name__ == "__main__":
    main()
