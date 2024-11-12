
import torch

from sklearn.metrics import precision_score, recall_score, f1_score








def train_model(model, train_loader, criterion, optimizer, num_epochs=25, writer=None, fold=0):
    """
    Entraîne le modèle sur les données d'entraînement.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            classification_loss = criterion(outputs, labels)
            loss = classification_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            print(
                f'Fold {fold}, Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Fold {fold}, Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        if writer:
            writer.add_scalar(f"Fold_{fold}/Train/Loss", epoch_loss, epoch)
    return model


def evaluate_model(model, val_loader, criterion, writer=None, fold=0):
    """
    Évalue le modèle sur les données de validation.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    val_loss = 0.0
    corrects = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            classification_loss = criterion(outputs, labels)
            loss = classification_loss
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    total_loss = val_loss / len(val_loader.dataset)
    accuracy = corrects.double() / len(val_loader.dataset)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    print(f'Fold {fold}, Validation Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
    if writer:
        writer.add_scalar(f"Fold_{fold}/Validation/Loss", total_loss, fold)
        writer.add_scalar(f"Fold_{fold}/Validation/Accuracy", accuracy, fold)
        writer.add_scalar(f"Fold_{fold}/Validation/Precision", precision, fold)
        writer.add_scalar(f"Fold_{fold}/Validation/Recall", recall, fold)
        writer.add_scalar(f"Fold_{fold}/Validation/F1", f1, fold)
    return total_loss, accuracy.item(), precision, recall, f1
