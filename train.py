import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from model import PlantDiseaseCNN
from dataset import get_dataloaders, get_plantdoc_dataloaders

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_npd_dir() -> Path:
    candidates = [
        PROJECT_ROOT
        / "New Plant Diseases Dataset(Augmented)"
        / "New Plant Diseases Dataset(Augmented)",
        Path(
            r"c:\Users\user\Deep_Learning_Analysis\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)"
        ),
    ]
    for path in candidates:
        if (path / "train").is_dir() and (path / "valid").is_dir():
            return path
    raise FileNotFoundError("NPD dataset not found. Place dataset under project root.")


def resolve_plantdoc_dir() -> Path:
    path = PROJECT_ROOT / "plantdoc"
    if (path / "train").is_dir() and (path / "test").is_dir():
        return path
    raise FileNotFoundError("PlantDoc dataset not found at plantdoc/train and plantdoc/test.")


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs=10,
    best_ckpt_path="best_model.pth",
    final_ckpt_path="final_model.pth",
):
    print(f"\nTraining started on device: {device}")

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == total_batches:
                print(
                    f"Epoch [{epoch+1:02d}/{num_epochs:02d}] | "
                    f"Batch [{batch_idx+1:04d}/{total_batches:04d}] | Loss: {loss.item():.4f}"
                )

        epoch_train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data).item()
                total += labels.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total

        print("=" * 65)
        print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}] Results:")
        print(
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}"
        )
        print("=" * 65 + "\n")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"--> Best model saved: {best_ckpt_path} (Val Acc: {best_val_acc:.4f})\n")

    torch.save(model.state_dict(), final_ckpt_path)
    print(f"Training finished. Final model saved: {final_ckpt_path}")
    print(f"Best Val Acc: {best_val_acc:.4f}")


def run_pretrain(args, device):
    """NPD pretrain (from scratch)."""
    npd_dir = resolve_npd_dir()
    print(f"Mode: NPD pretrain | data={npd_dir}")

    train_loader, val_loader = get_dataloaders(
        str(npd_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_level=args.augment_level,
        image_size=args.image_size,
    )

    model = PlantDiseaseCNN(num_classes=38).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        best_ckpt_path=args.best_ckpt,
        final_ckpt_path=args.final_ckpt,
    )


def run_finetune(args, device):
    """PlantDoc fine-tune (load NPD pretrained weights)."""
    plantdoc_dir = Path(args.plantdoc_dir) if args.plantdoc_dir else resolve_plantdoc_dir()
    pretrained_path = Path(args.pretrained)

    if not pretrained_path.is_file():
        raise FileNotFoundError(
            f"Pretrained weights not found: {pretrained_path}\n"
            "Run NPD pretrain first: python train.py --mode pretrain"
        )

    print(f"Mode: PlantDoc fine-tune")
    print(f"  Pretrained: {pretrained_path}")
    print(f"  PlantDoc:   {plantdoc_dir}")

    train_loader, test_loader, class_to_idx = get_plantdoc_dataloaders(
        str(plantdoc_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    model = PlantDiseaseCNN(num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    print(f"Loaded pretrained weights ({len(class_to_idx)} classes).")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        best_ckpt_path=args.best_ckpt,
        final_ckpt_path=args.final_ckpt,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plant disease CNN: NPD pretrain or PlantDoc fine-tune"
    )
    parser.add_argument(
        "--mode",
        choices=["pretrain", "finetune"],
        default="pretrain",
        help="pretrain: NPD only | finetune: PlantDoc with NPD weights",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument(
        "--augment-level",
        choices=["full", "minimal", "plantdoc"],
        default=None,
        help="NPD pretrain only (default: full)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="best_model.pth",
        help="NPD checkpoint for finetune mode",
    )
    parser.add_argument(
        "--plantdoc-dir",
        type=str,
        default=None,
        help="PlantDoc root (train/ + test/). Default: ./plantdoc",
    )
    parser.add_argument("--best-ckpt", type=str, default=None)
    parser.add_argument("--final-ckpt", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.epochs is None:
        args.epochs = 10 if args.mode == "pretrain" else 15
    if args.lr is None:
        args.lr = 1e-3 if args.mode == "pretrain" else 1e-4
    if args.augment_level is None:
        args.augment_level = "full" if args.mode == "pretrain" else "plantdoc"
    if args.best_ckpt is None:
        args.best_ckpt = (
            "best_model.pth" if args.mode == "pretrain" else "best_model_plantdoc.pth"
        )
    if args.final_ckpt is None:
        args.final_ckpt = (
            "final_model.pth" if args.mode == "pretrain" else "final_model_plantdoc.pth"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.mode == "pretrain":
        run_pretrain(args, device)
    else:
        run_finetune(args, device)


if __name__ == "__main__":
    main()
