from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

# import torchaudio
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from typing import Iterable
import librosa
import numpy as np


class UrbanSoundDataset(Dataset):
    def __init__(
        self, csv_path: Path, file_path: Path, folderList: Iterable[int], new_sr=8000, audio_len=20, sampling_ratio=5
    ):
        """[summary]

        Args:
            csv_path (Path): Path to dataset metadata csv
            file_path (Path): Path to data folders
            folderList (Iterable[int]): Data folders to be included in dataset
            new_sr (int, optional): New sampling rate. Defaults to 8000.
            audio_len (int, optional): Audio length based on new sampling rate (sec). Defaults to 20.
            sampling_ratio (int, optional): Additional downsampling ratio. Defaults to 5.
        """

        df = pd.read_csv(csv_path)
        self.file_names = []
        self.labels = []
        self.folders = []
        for i in range(0, len(df)):
            if df.iloc[i, 5] in list(folderList):
                self.labels.append(df.iloc[i, 6])
                self.folders.append(df.iloc[i, 5])
                temp = "fold" + str(df.iloc[i, 5]) + "/" + str(df.iloc[i, 0])
                temp = file_path / temp
                self.file_names.append(temp)

        self.file_path = Path(file_path)
        self.folderList = folderList
        self.new_sr = new_sr
        self.audio_len = audio_len
        self.sampling_ratio = sampling_ratio

    def __getitem__(self, index):
        # format the file path and load the file
        path = self.file_names[index]
        sound, sr = librosa.core.load(str(path), mono=False, sr=None)
        if sound.ndim < 2:
            sound = np.expand_dims(sound, axis=0)
        # Convert into single channel format
        sound = sound.mean(axis=0, keepdims=True)
        # Downsampling
        sound = librosa.core.resample(sound, orig_sr=sr, target_sr=self.new_sr)

        # Zero padding to keep desired audio length in seconds
        const_len = self.new_sr * self.audio_len
        tempData = np.zeros([1, const_len])
        if sound.shape[1] < const_len:
            tempData[0, : sound.shape[1]] = sound[:]
        else:
            tempData[0, :] = sound[0, :const_len]
        sound = tempData
        # Resampling
        new_const_len = const_len // self.sampling_ratio
        soundFormatted = torch.zeros([1, new_const_len])
        soundFormatted[0, :] = torch.tensor(sound[0, ::5], dtype=float)

        return soundFormatted, self.labels[index]

    def __len__(self):
        return len(self.file_names)


class NetM3(nn.Module):
    def __init__(self):
        super(NetM3, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, 80, 4)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128, 128, 3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(128, 256, 3)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(256, 512, 3)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(30)  # input should be 512x30 so this outputs a 512x1
        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = self.avgPool(x)
        x = x.permute(0, 2, 1)  # change the 512x1 to 1x512
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)  # Output: torch.Size([N, 1, 10])


def train(model, epoch, train_loader, device, optimizer, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)
        output = output.permute(1, 0, 2)  # original output dimensions are batchSizex1x10
        loss = F.nll_loss(output[0], target)  # the loss functions expects a batchSizex10 input
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100.0 * batch_idx / len(train_loader), loss
                )
            )


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output.permute(1, 0, 2)[0]
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), accuracy
        )
    )
    return test_loss, accuracy


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NetM3()
    if torch.cuda.device_count() > 1:
        print("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def save_model(model, model_dir):
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), path)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128, help="train batch size")
    parser.add_argument(
        "--test-batch-size", type=int, default=2000, help="test batch size",
    )
    parser.add_argument("--epochs", type=int, default=2, help="number of epochs (default: 2)")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.1, help="Learning rate step gamma")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="Optimizer regularization")
    parser.add_argument("--stepsize", type=int, default=5, help="Step LR size")
    parser.add_argument("--model", type=str, default="m3")
    parser.add_argument("--num-workers", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--cv", type=int, default=0, help="0: No cross validation 1: with cross validation")

    # Container environment
    parser.add_argument("--model-dir", type=str, default=os.getenv("SM_MODEL_DIR", "./"))
    if os.getenv("SM_HOSTS") is not None:
        # parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
        parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
        # parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    args = parser.parse_args()
    print(args)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # On SageMaker
    if os.getenv("SM_HOSTS") is not None:
        print("Running on sagemaker")
        datapath = Path(args.data_dir)
        csvpath = datapath / "UrbanSound8K.csv"
        print("datapath", datapath)
        print("csvpath", csvpath)
    # Local
    else:
        print("Running on local")
        full_filepath = Path(__file__).resolve()
        parent_path = full_filepath.parent.parent
        datapath = parent_path / "data/UrbanSound8K"
        csvpath = datapath / "UrbanSound8K.csv"

    kwargs = {"num_workers": args.num_workers, "pin_memory": True} if torch.cuda.is_available() else {}
    print(kwargs)

    # 10 fold cross validation
    all_scores = []
    if args.cv == 1:
        print("Cross validation: Enable")
        cv = 10
    else:
        cv = 1

    for i in range(1, cv + 1):
        folders = list(range(1, 11))
        test_folder = [i]
        train_folder = set(folders) - set([i])
        print(f"***** Processing fold({i}) *****")

        train_set = UrbanSoundDataset(csvpath, datapath, train_folder)
        test_set = UrbanSoundDataset(csvpath, datapath, test_folder)
        print("Training size:", len(train_set))
        print("Test size:", len(test_set))

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        print("Loading model:", args.model)
        if args.model == "m3":
            model = NetM3()
        else:
            model = NetM3()

        if torch.cuda.device_count() > 1:
            print("There are {} gpus".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)

        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

        log_interval = args.log_interval

        for epoch in range(1, args.epochs + 1):
            print("Learning rate:", scheduler.get_last_lr()[0])
            train(model, epoch, train_loader, device, optimizer, log_interval)
            loss, accuracy = test(model, test_loader, device)
            scheduler.step()

        print(f"Accuracy for fold ({i}): {accuracy}")
        all_scores.append(accuracy)

    print(f"Final score: {sum(all_scores)/len(all_scores):.2f}%")

    # Save Model
    save_model(model, args.model_dir)


if __name__ == "__main__":
    main()
