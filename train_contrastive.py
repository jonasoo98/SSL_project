import torch
import time

import seaborn as sns
import numpy as np

from torch.utils.data import  DataLoader
from torchvision import datasets, transforms

from dataset import ContrastiveDataset
from optimizer import LARS
from loss import NT_Xent
from model import ContrastiveLearningModel

import matplotlib.pyplot as plt

def main(): 

    # Basic setup stuff
    sns.set_theme()
    seed = 42 
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    ssl_batch_size = 200
    num_workers = 0 # means no sub-processes, needed for debugging
    train_dataloader = DataLoader(
        train_dataset, batch_size=ssl_batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=ssl_batch_size, shuffle=False, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=ssl_batch_size, shuffle=False, num_workers=num_workers
    )

    # Creating model, optimizer and loss function
    model = ContrastiveLearningModel().to(device)
    optimizer = LARS(
        [params for params in model.parameters() if params.requires_grad],
        lr=0.2,
        weight_decay=1e-6,
        exclude_from_weight_decay=["batch_normalization", "bias"],
    )
    criterion = NT_Xent(batch_size=ssl_batch_size, temperature=0.5)

    # Running training
    num_epochs = 40

    training_losses = []
    validation_losses = []
    for epoch in range(num_epochs): 

    start = time.time()
    model.train()
    training_loss = 0
    for (x_i, x_j) in train_dataloader: 
        optimizer.zero_grad()
        x_i, x_j = x_i.to(device), x_j.to(device)

        z_i = model(x_i)
        z_j = model(x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()
        training_loss += loss.item()
    
    training_loss /= len(train_dataloader)
    training_losses.append(training_loss)
    
    model.eval()
    with torch.no_grad(): 
        validation_loss = 0
        for (x_i, x_j) in val_dataloader: 
        x_i, x_j = x_i.to(device), x_j.to(device)

        z_i = model(x_i)
        z_j = model(x_j)

        loss = criterion(z_i, z_j)
        validation_loss += loss.item()

        validation_loss /= len(val_dataloader)
        validation_losses.append(validation_loss)
        
        end = time.time()
        
    print(f"Epoch #{epoch+1}, training loss: {training_loss}, validation loss: {validation_loss}, time: {end - start:.2f}")

    # Plotting results
    x = np.linspace(1, len(training_losses), len(training_losses))
    plt.plot(x, training_losses, label="Training")
    plt.plot(x, validation_losses, label="Valdation")

    plt.xlabel("Num epochs")
    plt.ylabel("Loss")
    plt.title("Contrastive Loss")
    plt.legend()

    plt.savefig("contrastive_loss.png")
    plt.close()

    # Saving the model to file
    model_path = "models/encoder.pth" 
    torch.save(model.encoder, model_path)

if __name__ == "__main__": 
    main()