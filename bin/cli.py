import typer

from ser.train import my_train

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
def infer():
    print("This is where the inference code will go")
