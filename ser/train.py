import torch
from torch import optim
import torch.nn.functional as F
from dataclasses import dataclass
from datetime import datetime

from ser.model import Net
from ser.constants import PROJECT_ROOT, DATA_DIR, RESULTS_DIR
from ser.data import load_data
from ser.transforms import get_transforms

import json

def my_train(name: str, ):

    print(f"Running experiment {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 2
    batch_size = 1000
    learning_rate = 0.01

    # ----- TO-DO: save the parameters! -----
    @dataclass
    class Params:
        epochs: int 
        bs: int
        lr: int

    params = Params(epochs, bs, lr)
    timestamp = datetime.now(tz=None)
    filename =  RESULTS_DIR / f"params_{name}_{timestamp}.json"
    with open(filename, 'w') as file_object:  
        json.dump(params, file_object)

    # load model
    model = Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # torch transforms
    ts = get_transforms()

    # dataloaders
    training_dataloader = load_data(directory=DATA_DIR, download_bool=True, train_bool=True, ts=ts, batch_size=batch_size, shuffle_bool=True, num_workers=1)
    validation_dataloader = load_data(directory=DATA_DIR, download_bool=True, train_bool=False, ts=ts, batch_size=batch_size, shuffle_bool=False, num_workers=1)
  

    # train
    val_acc_old = 0    # initialise val accuracy to save best model
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(training_dataloader):
            images, labels = images.to(device), labels.to(device)
            model.train()
            optimizer.zero_grad()
            output = model(images)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
            print(
                f"Train Epoch: {epoch} | Batch: {i}/{len(training_dataloader)} "
                f"| Loss: {loss.item():.4f}"
            )

        # validate
        val_loss = 0
        correct = 0

        with torch.no_grad():
            for images, labels in validation_dataloader:
                images, labels = images.to(device), labels.to(device)
                model.eval()
                output = model(images)
                val_loss += F.nll_loss(output, labels, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
            val_loss /= len(validation_dataloader.dataset)
            val_acc = correct / len(validation_dataloader.dataset)

            print(
                f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}"
            )

        if val_acc > val_acc_old:
            directory = RESULTS_DIR / f'model_{name}_{timestamp}.pt'
            torch.save(model, directory)

        val_acc_old = val_acc

