import argparse
import os.path
import pandas as pd
import numpy as np
from ProteinGraphDataset import ProteinGraphDataset
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.loader import DataLoader
from PeptiTox import PeptiTox
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             average_precision_score, matthews_corrcoef,
                             confusion_matrix)
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_

warnings.filterwarnings("ignore")


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calculate_metrics(y_true, y_pred, y_prob):
    metrics = {}

    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['F1'] = f1_score(y_true, y_pred)
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)

    metrics['AUROC'] = roc_auc_score(y_true, y_prob)
    metrics['AUPRC'] = average_precision_score(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['Sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['FPR'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    metrics['FDR'] = fp / (fp + tp) if (fp + tp) > 0 else 0.0

    return metrics


def format_metrics(metrics):
    return " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])


def save_results(fold_results, filename="results.csv"):
    df = pd.DataFrame(fold_results)

    summary = pd.DataFrame({
        'Mean': df.mean(),
        'Std': df.std()
    }).T

    final_df = pd.concat([df, summary])

    final_df = final_df.applymap(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

    if not os.path.exists('results'):
        os.makedirs('results')

    final_df.to_csv(f"results/{filename}", index=False)
    print(f"\nResults saved to results/{filename}")


def train(model, loader, optimizer, criterion, device, clip_grad=False):
    model.train()
    total_loss = 0.0
    total_norm = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(X=batch.X, h_V=batch.node_feat, edge_index=batch.edge_index, batch_id=batch.batch)
        # logits = output.view(-1)
        labels = batch.y.float()
        loss = criterion(output, labels)
        loss.backward()

        if clip_grad:
            total_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
        # print(f"Grad Norm: {total_norm:.4f}")

        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(X=batch.X, h_V=batch.node_feat, edge_index=batch.edge_index, batch_id=batch.batch)
            labels = batch.y.float()

            loss = criterion(output, labels)
            total_loss += loss.item() * batch.num_graphs

            probs = torch.sigmoid(output).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            # correct += (preds == labels.long()).sum().item()
            # total += labels.size(0)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().astype(int))
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    return total_loss / len(loader.dataset), metrics


def main(args):
    data = pd.read_csv(os.path.join(args.data_path, 'merged_data.csv'))
    sequences = data['sequence'].values
    labels = data['label'].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    best_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(sequences, labels)):
        print(f"\nFold {fold + 1}/5")

        # train_idx, val_idx = train_idx[:args.batch_size * 11], val_idx[:args.batch_size * 11]
        train_seq, val_seq = sequences[train_idx], sequences[val_idx]
        train_label, val_label = labels[train_idx], labels[val_idx]

        train_dataset = ProteinGraphDataset(train_seq, train_label, args.data_path)
        val_dataset = ProteinGraphDataset(val_seq, val_label, args.data_path)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=4)

        model = PeptiTox(
            node_input_dim=args.node_input_dim,
            edge_input_dim=args.edge_input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            augment_eps=args.augment_eps,
            dropout=args.dropout
        ).to(args.device)

        # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 50

        best_auc = 0
        fold_metrics = []

        for epoch in range(args.epochs):
            train_loss = train(model, train_loader, optimizer, criterion, args.device, clip_grad=False)
            val_loss, val_metrics = validate(model, val_loader, criterion, args.device)

            scheduler.step(val_loss)

            save_path = f"saved_models/{args.model}/batch_{args.batch_size}_lr_{args.lr}_wd_{args.weight_decay}_dp_{args.dropout}_layer_{args.num_layers}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # if val_metrics['AUROC'] > best_auc:
            #     best_auc = val_metrics['AUROC']
            #     best_epoch_metrics = {
            #         'Fold': fold+1,
            #         'Epoch': epoch+1,
            #         **val_metrics
            #     }
            #     torch.save(model.state_dict(), f"saved_models/fold{fold+1}_best_model.pth")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_epoch_metrics = {
                    'Fold': fold + 1,
                    'Epoch': epoch + 1,
                    **val_metrics
                }
                torch.save(model.state_dict(), f"{save_path}/fold{fold + 1}.pth")
                print(f"Saved model at epoch {epoch + 1}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            print(f"Epoch {epoch + 1}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"{format_metrics(val_metrics)}")

        fold_results.append(best_epoch_metrics)

        print(f"\nFold {fold + 1} Best Metrics:")
        print(format_metrics(best_epoch_metrics))

    save_results(pd.DataFrame(fold_results),
                 f"{args.model}_batch_{args.batch_size}_lr_{args.lr}_wd_{args.weight_decay}_dp_{args.dropout}_layer_{args.num_layers}.csv")

    print("\nCross Validation Final Results:")
    final_metrics = pd.DataFrame(fold_results).mean().to_dict()
    print(format_metrics(final_metrics))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./Data/')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--node_input_dim', type=int, default=1280 + 184)
    parser.add_argument('--edge_input_dim', type=int, default=450)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--augment_eps', type=float, default=0.1)
    parser.add_argument('--model', type=str, default='Transformer')
    args = parser.parse_args()
    seed_everything(42)
    main(args)
