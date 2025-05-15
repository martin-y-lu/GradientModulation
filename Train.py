from collections import defaultdict, deque
from tqdm import tqdm
import numpy as np
import os
import torch

def train_and_evaluate(network, train_loader, val_loader, epochs=10, lr=0.001, log_every=100, device='cpu',
                       save_path = None, save_every = 5,
                       logger = None):
    model = network.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    logs = defaultdict(list)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        grad_norms = []
        weight_updates = []
        
        running_loss = deque(maxlen=50)  # For smooth display
        running_acc = deque(maxlen=50)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradients and updates
            total_norm = 0
            weight_update_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
                    weight_update_norm += (lr * p.grad.data).norm(2).item() ** 2

            grad_norms.append(np.sqrt(total_norm))
            weight_updates.append(np.sqrt(weight_update_norm))

            optimizer.step()

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size

            batch_acc = (outputs.argmax(1) == labels).float().mean().item()
            correct += (outputs.argmax(1) == labels).sum().item()

            if batch_idx % log_every == 0:
                logs['batches'].append({
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'loss': loss.item(),
                    'acc': batch_acc,
                    'grad_norm': grad_norms[-1],
                    'weight_update_norm': weight_updates[-1],
                    'samples_seen': total
                })

            # Track running stats for display
            running_loss.append(loss.item())
            running_acc.append(batch_acc)

            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f'{np.mean(running_loss):.4f}',
                    'acc': f'{np.mean(running_acc) * 100:.2f}%',
                    'grad_norm': f'{grad_norms[-1]:.2f}'
                })

        logs['train_loss'].append(total_loss / total)
        logs['train_acc'].append(correct / total)
        logs['train_grad_norm'].append(np.mean(grad_norms))
        logs['train_weight_update_norm'].append(np.mean(weight_updates))

        val_loss, val_acc = validate_model(model,val_loader,device)

        logs['val_loss'].append(val_loss)
        logs['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {logs['train_loss'][-1]:.4f} | "
              f"Train Acc: {logs['train_acc'][-1]*100:.2f}% | "
              f"Val Acc: {logs['val_acc'][-1]*100:.2f}%")

        if save_path != None and (epoch % save_every == 0):
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), f"{save_path}/model_epoch_{epoch}.pt")
            
    if save_path != None:
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), f"{save_path}/model_final.pt")
    
    if logger != None:
        logger(("gmodrelu",l,k),logs)
        
    return logs, model

def validate_model(model, val_loader,device, criterion = torch.nn.CrossEntropyLoss()):
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += inputs.size(0)

    return val_loss / val_total , val_correct / val_total