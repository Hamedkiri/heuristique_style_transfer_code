import os
import torch

from sklearn.metrics import precision_score, recall_score


torch.autograd.set_detect_anomaly(True)  # Pour activer la détection d'anomalies







def load_model(model, model_path, device):
    if os.path.isfile(model_path):
        print(f"Loading pre-trained ResNet50 model from {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully.")
    else:
        raise FileNotFoundError(f"No model found at {model_path}")


def save_model_weights(model, save_path):
    state_dict = {
        'truncated_encoder': model.truncated_encoder.state_dict(),
        'classifier': model.classifier.state_dict(),
        'attention': model.attention.state_dict()  # Ajouter les poids de l'attention
    }
    torch.save(state_dict, save_path)
    print(f"Model weights saved to {save_path}")



def load_model_weights(model, load_path):
    if os.path.isfile(load_path):
        state_dict = torch.load(load_path, map_location=model.device)

        try:
            if 'truncated_encoder' in state_dict:
                model.truncated_encoder.load_state_dict(state_dict['truncated_encoder'], strict=True)
            else:
                print("Warning: 'truncated_encoder' not found in state_dict.")

            if 'classifier' in state_dict:
                model.classifier.load_state_dict(state_dict['classifier'], strict=True)
            else:
                print("Warning: 'classifier' not found in state_dict.")

            if 'attention' in state_dict:  # Charger les poids de l'attention
                model.attention.load_state_dict(state_dict['attention'], strict=True)
            else:
                print("Warning: 'attention' not found in state_dict.")

            print(f"Model weights loaded from {load_path} using direct method.")

        except (KeyError, RuntimeError) as e:
            print(f"Direct loading failed with error: {e}")
            print("Attempting to load weights by processing keys...")

            truncated_encoder_state_dict = {}
            classifier_state_dict = {}
            attention_state_dict = {}

            for key, value in state_dict.items():
                if key.startswith('truncated_encoder'):
                    new_key = key.replace('truncated_encoder.', '')
                    truncated_encoder_state_dict[new_key] = value
                elif key.startswith('classifier'):
                    new_key = key.replace('classifier.', '')
                    classifier_state_dict[new_key] = value
                elif key.startswith('attention'):
                    new_key = key.replace('attention.', '')
                    attention_state_dict[new_key] = value

            model.truncated_encoder.load_state_dict(truncated_encoder_state_dict, strict=True)
            model.classifier.load_state_dict(classifier_state_dict, strict=True)
            model.attention.load_state_dict(attention_state_dict, strict=True)

            print(f"Model weights loaded from {load_path} by processing keys.")
    else:
        print(f"No weights file found at {load_path}. Proceeding without loading weights.")


def set_parameter_requires_grad(model, freeze_encoder):
    if freeze_encoder:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                print(f"Layer {name} is unfrozen.")


def train_model(model, train_loader, criterion, optimizer, num_epochs=25, writer=None, fold=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            # Debugging : vérifier les dimensions des sorties et des labels
            print(f"outputs.shape: {outputs.shape}, labels.shape: {labels.shape}")

            loss = criterion(outputs, labels)
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
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    total_loss = val_loss / len(val_loader.dataset)
    accuracy = corrects.double() / len(val_loader.dataset)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    print(
        f'Fold {fold}, Validation Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
    if writer:
        writer.add_scalar(f"Fold_{fold}/Validation/Loss", total_loss)
        writer.add_scalar(f"Fold_{fold}/Validation/Accuracy", accuracy)
        writer.add_scalar(f"Fold_{fold}/Validation/Precision", precision)
        writer.add_scalar(f"Fold_{fold}/Validation/Recall", recall)
    return total_loss, accuracy.item(), precision, recall
