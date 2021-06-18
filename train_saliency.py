#!/usr/bin/evn python3

import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple
import scipy
from scipy import spatial, stats
import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import argparse
from pathlib import Path
from dataset import Salicon
import visualisation
import evaluation
import pickle

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a simple CNN on Saliency Prediction",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=3e-2, type=float, help="Learning rate")
parser.add_argument(
    "--batch-size",
    default=128,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=1000,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)


class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def main(args):
    transform = transforms.ToTensor()
    args.dataset_root.mkdir(parents=True, exist_ok=True)
    train_dataset = Salicon("train.pkl")
    test_dataset = Salicon("val.pkl") # change to test.pkl

    # train_dataset =  pickle.load(open("train.pkl", 'rb')) #dataset.Salicon("train.pkl")
    # test_dataset = pickle.load(open("val.pkl", 'rb')) #dataset.Salicon("val.pkl")
    # train_dataset = transform(train_dataset.dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    model = CNN(height=96, width=96, channels=3) # class_count=10) # class count chnage?

    ## TASK 8: Redefine the criterion to be softmax cross entropy
    #criterion = lambda logits, labels: torch.tensor(0)
    criterion = nn.MSELoss()

    ## TASK 11: Define the optimizer
    optimizer = optim.SGD(model.parameters(), args.learning_rate, weight_decay=0.0005, momentum=0.9, nesterov=True)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    trainer = Trainer(
        model, train_loader, test_loader, criterion, optimizer, summary_writer, DEVICE
    )

    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    summary_writer.close()


class CNN(nn.Module):
    def __init__(self, height: int = 96, width: int = 96, channels: int = 3): #, class_count: int = 10): # change class count
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        #self.class_count = class_count

        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            kernel_size=(5, 5),
            padding=(2, 2),
        )
        self.initialise_layer(self.conv1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        ## TASK 2-1: Define the second convolutional layer and initialise its parameters
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.initialise_layer(self.conv2)

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.initialise_layer(self.conv3)

        self.conv4 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.initialise_layer(self.conv4)

        ## TASK 3-1: Define the second pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        ## TASK 5-1: Define the first FC layer and initialise its parameters
        self.fcl1 = nn.Linear(30976, 4608)
        self.initialise_layer(self.fcl1)
        ## TASK 6-1: Define the last FC layer and initialise its parameters
        self.fcl2 = nn.Linear(2304, 2304)
        self.initialise_layer(self.fcl2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(images))
        x = self.pool1(x)
        ## TASK 2-2: Pass x through the second convolutional layer
        x = F.relu(self.conv2(x))
        ## TASK 3-2: Pass x through the second pooling layer
        x = self.pool2(x)
        ## TASK 4: Flatten the output of the pooling layer so it is of shape
        ##         (batch_size, 4096)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool3(x)

        x = torch.flatten(x, start_dim=1)
        ## TASK 5-2: Pass x through the first fully connected layer
        x = self.fcl1(x) # output 128, 4608
        slice1 = x.narrow(1, 0, 2304)
        slice2 = x.narrow(1, 2304, 2304)
        # maxout
        x = torch.max(slice1, slice2)
        ## TASK 6-2: Pass x through the last fully connected layer
        x = self.fcl2(x)
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            #nn.init.zeros_(layer.bias) # 
            nn.init.constant_(layer.bias, 0.1)
        if hasattr(layer, "weight"):
            #nn.init.kaiming_normal_(layer.weight) # 
            nn.init.normal_(layer.weight, 0, 0.01)



class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        lrs= np.linspace(0.03, 0.0001, epochs+1)
        self.model.train()
        final_preds=0
        gts=0
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for batch, labels in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()


                ## TASK 1: Compute the forward pass of the model, print the output shape
                ##         and quit the program
                #output =

                ## TASK 7: Rename `output` to `logits`, remove the output shape printing
                ##         and get rid of the `import sys; sys.exit(1)`
                logits = self.model.forward(batch)
                ## TASK 9: Compute the loss using self.criterion and
                ##         store it in a variable called `loss`
                loss = torch.sqrt(self.criterion(logits, labels))
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                ## TASK 10: Compute the backward pass

                ## TASK 12: Step the optimizer and then zero out the gradient buffers.

                with torch.no_grad():
                    preds = logits
                    final_preds = preds
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            for g in self.optimizer.param_groups:
                g['lr'] = lrs[epoch+1]

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()

        # pickle_out
        #print(final_preds.shape)
        #print(gts.shape)
        #pickle.dump(final_preds, open("preds.pkl", "wb"))
        #pickle.dump(gts, open("gts.pkl", "wb"))


    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def validate(self):
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch, labels in self.val_loader: # test_loader
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                testlabels = labels.cpu().numpy()
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))
        pickle.dump(np.array(results["labels"]), open("final_label.pkl", "wb"))
        pickle.dump(np.array(results["preds"]), open("final_preds.pkl", "wb"))

        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        average_loss = total_loss / len(self.val_loader)

        self.summary_writer.add_scalars(
                "accuracy",
                {"test": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")

def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    len1 = labels.shape
    len1 = len1[0] * len1[1]
    len2 = preds.shape
    len2 = len2[0] * len2[1]
    assert len1 == len2
    #train_acc = np.where(preds == labels)
    #len3 = train_acc.shape
    #len3 = len3[0] * len3[1]
    #train_acc = torch.sum(preds == labels)
    #print(len3)
  #  count = 0
  #  for i in range(128):
  #    for j in range(2304):
  #      if (labels[i,j] == preds[i,j]):
  #        count = count + 1
    return float((labels == preds).sum()) / len1
    #return len3 / len1


def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = f'CNN_bs={args.batch_size}_lr={args.learning_rate}_run_'
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)




if __name__ == "__main__":
    main(parser.parse_args())
