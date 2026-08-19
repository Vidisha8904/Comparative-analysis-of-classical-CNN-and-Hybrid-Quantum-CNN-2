"""Training and evaluation loops shared by both the classical and hybrid runs."""

import os
import time

import torch


def save_checkpoint(model, optimizer, epoch, batch_idx, seed, path):
    """Persist model/optimizer state and the training progress for resume support."""
    checkpoint_dir = os.path.dirname(path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "batch_idx": batch_idx,
        "seed": seed,
    }
    torch.save(payload, path)
    return payload


def load_checkpoint(path):
    """Load a checkpoint if it exists; otherwise return None."""
    if not path or not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            n_batches += 1
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    avg_loss = total_loss / n_batches
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    return avg_loss, y_pred, y_true


def run_training(model, train_loader, test_loader, config, logger):
    """Run the full training loop over config['training']['epochs'], logging each epoch."""
    from src.training.metrics import compute_metrics, count_parameters

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_cfg = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()

    checkpoint_every_n_batches = train_cfg.get("checkpoint_every_n_batches")
    resume_from_checkpoint = train_cfg.get("resume_from_checkpoint")
    checkpoint_path = os.path.join(config["output_dir"], "checkpoints", "checkpoint_latest.pt")
    max_batches = train_cfg.get("max_batches")

    start_epoch = 1
    batch_resume_offset = 0

    if resume_from_checkpoint:
        checkpoint = load_checkpoint(resume_from_checkpoint)
        if checkpoint is None:
            raise FileNotFoundError(f"Checkpoint not found at {resume_from_checkpoint}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 1)
        batch_resume_offset = checkpoint.get("batch_idx", -1) + 1

    total_epochs = train_cfg["epochs"]
    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            if batch_idx < batch_resume_offset and epoch == start_epoch:
                continue
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            running_loss += batch_loss
            if (batch_idx + 1) % 50 == 0 or batch_idx + 1 == len(train_loader):
                elapsed = time.perf_counter() - epoch_start
                print(
                    f"Epoch {epoch}/{total_epochs} batch {batch_idx + 1}/{len(train_loader)} "
                    f"loss={batch_loss:.4f} running_loss={running_loss / (batch_idx + 1):.4f} "
                    f"elapsed={elapsed:.2f}s"
                )

            if checkpoint_every_n_batches and (batch_idx + 1) % checkpoint_every_n_batches == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    seed=config["training_seed"],
                    path=checkpoint_path,
                )

            if max_batches and batch_idx + 1 >= max_batches:
                break

        batch_resume_offset = 0

        train_loss = running_loss / max(1, len(train_loader))
        test_loss, y_pred, y_true = evaluate(model, test_loader, criterion, device)
        epoch_time = time.perf_counter() - epoch_start

        metrics_dict = compute_metrics(y_true, y_pred)
        metrics_dict["parameter_count"] = count_parameters(model)
        logger.log_epoch(epoch, train_loss, test_loss, metrics_dict, epoch_time)

        save_checkpoint(
            model,
            optimizer,
            epoch=epoch,
            batch_idx=len(train_loader) - 1,
            seed=config["training_seed"],
            path=checkpoint_path,
        )

        print(
            f"Epoch {epoch}/{total_epochs} "
            f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} "
            f"acc={metrics_dict['accuracy']:.4f} time={epoch_time:.2f}s"
        )

    return model
