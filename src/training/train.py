from datetime import datetime
import os
import sys
from rich import print
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.datasets.palm_dataset import PalmDataset
from src.model.palm_net import PalmNet
from src.transforms.transform_pipeline import train_transform, eval_transform

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="palmnet")
    parser.add_argument("--save_dir", "-s", type=str, default="../../models/")
    parser.add_argument("--runs_dir", "-r", type=str, default="../../runs/")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", "-b", type=int, default=64)
    parser.add_argument("--epochs", "-e", type=int, default=200)
    parser.add_argument("--num_workers", "-n", type=int, default=4)
    parser.add_argument("--margin", "-m", type=float, default=0.5)
    parser.add_argument("--patience", "-p", type=int, default=10)
    parser.add_argument("--factor", type=float, default=0.1, help="LR will be multiplied by this factor")
    parser.add_argument("--lr_patience", type=int, default=5, help="Number of epochs with no improvement before reducing LR")
    parser.add_argument("--colour", "-c", type=str, default="cyan")

    return parser.parse_args()

def train(device, writer, args):
    train_dataset = PalmDataset(root=args.train_path, transform=train_transform)
    val_dataset = PalmDataset(root=args.val_path, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    model = PalmNet().to(device)
    criterion = nn.TripletMarginLoss(margin=args.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Theo dõi validation loss, ko cải thiện trong args.lr_patience epoch, lr giảm đi 1/args.factor lần
    scheduler = ReduceLROnPlateau(optimizer, mode= "min", factor=args.factor, patience=args.lr_patience)

    best_val_loss = float("inf")  # dương vô cực
    epochs_no_improve = 0

    num_epochs = args.epochs
    num_iter = len(train_loader)

    pos_sum = 0
    neg_sum = 0
    count = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, colour=args.colour, file=sys.stdout)
        for i, (anchor, positive, negative) in enumerate(pbar):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            dist_ap = torch.norm(emb_a - emb_p, dim=1)
            dist_an = torch.norm(emb_a - emb_n, dim=1)
            count += 1

            pos_sum += dist_ap.mean().item()
            neg_sum += dist_an.mean().item()

            loss = criterion(emb_a, emb_p, emb_n)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            writer.add_scalar("Loss/train", loss.item(), epoch * num_iter + i)
            pbar.set_description(f"Epoch {epoch}/{num_epochs} | Loss: {loss.item():.4f}")

        train_loss /= num_iter

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)

                emb_a = model(anchor)
                emb_p = model(positive)
                emb_n = model(negative)

                loss = criterion(emb_a, emb_p, emb_n)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"[yellow]Val Loss: {val_loss:.4f}, Train Loss: {train_loss:.4f}, Pos mean: {pos_sum / count:.4f}, Neg mean: {neg_sum / count:.4f}[/yellow]")
        writer.add_scalar("Loss/val", val_loss, epoch)

        scheduler.step(val_loss) # lr Scheduler

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_path = os.path.join(args.save_dir, f"{args.model_name}.pth")
            torch.save(model.state_dict(), save_path)
            print("[green]Best model saved[/green]")
        else:
            epochs_no_improve += 1
        # Dừng sau args.patience epoch nếu validation loss ko cải thiện
        if epochs_no_improve >= args.patience:
            print("[bold red]Early stopping triggered[/bold red]")
            break

if __name__ == '__main__':
    args = get_args()

    os.makedirs(args.save_dir, exist_ok=True) # tạo thư mục lưu model nếu chưa tồn tại

    base_dir = args.runs_dir
    os.makedirs(base_dir, exist_ok=True) # tạo thư mục lưu tensorboard nếu chưa tồn tại
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(base_dir, f"exp_{timestamp}")
    writer = SummaryWriter(log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[blue]Device: {device}[/blue]")

    train(device, writer, args)