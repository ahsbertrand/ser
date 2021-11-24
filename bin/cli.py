import typer
<<<<<<< Updated upstream

from ser.train import my_train
=======
import torch
import git
import json

from ser.train import train as run_train
from ser.constants import RESULTS_DIR
from ser.data import train_dataloader, val_dataloader, test_dataloader
from ser.params import Params, save_params
from ser.transforms import transforms, normalize
from ser.infer import run_inference
from ser.utils import generate_ascii_art
>>>>>>> Stashed changes

main = typer.Typer()

@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    learning_rate: str = typer.Option(
        ..., "--lr", help="Learning rate."
    ),
    batch_size: str = typer.Option(
        ..., "--bs", help="Batch size."
    ),
    epochs: str = typer.Option(
        ..., "--epochs", help="Number of epochs"
    ),
):
    my_train(name, learning_rate, epochs, batch_size)
    


@main.command()
<<<<<<< Updated upstream
def infer():
    print("This is where the inference code will go")
=======
def infer(
    run_path: Path
):
    # class label in MNIST dataset
    label = 6

    # TO-DO load the parameters from the run_path so we can print them out!

    # select image to run inference for
    dataloader = test_dataloader(1, transforms(normalize))
    images, labels = next(iter(dataloader))
    while labels[0].item() != label:
        images, labels = next(iter(dataloader))

    # load the model
    model = torch.load(run_path / "model.pt")

    # perform inference
    pixels, pred, certainty = run_inference(model, images)

    print(generate_ascii_art(pixels))

    # print results 
    with open(run_path / "params.json") as f:
        params = json.load(f)
        print(f"Experiment name       : {params['name']}")
        print(f"Number of epochs      : {params['epochs']}")
        print(f"Batch size            : {params['batch_size']}")
        print(f"Learning rate         : {params['learning_rate']}")
    
    print(f"Predicted number is   : {pred}")
    print(f"Prediction confidence : {certainty}")
>>>>>>> Stashed changes
