import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn, optim
from torchvision.models import efficientnet_b4
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

# ======================
# Configuration
# ======================
METADATA_PATH = "../notebooks/data/HAV_DF/video_metadata.csv"
TRAIN_FRAMES_ROOT = "../notebooks/data/processed_data/train/frames"
TEST_FRAMES_ROOT = "../notebooks/data/processed_data/test/frames"
BATCH_SIZE = 32
IMAGE_SIZE = (380, 380)  # EfficientNet-B4 input size
NUM_EPOCHS = 10
NUM_CLASSES = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

# ======================
# Dataset Preparation
# ======================
class DeepFakeFrameDataset(Dataset):
    def __init__(self, frames_root, metadata_df, transform=None):
        self.samples = []
        self.transform = transform
        self.label_map = {'REAL': 0, 'FAKE': 1}
        
        # Create video to label mapping (remove .mp4 extension)
        video_to_label = {}
        for _, row in metadata_df.iterrows():
            video_name = os.path.splitext(row['video_name'])[0]
            video_to_label[video_name] = self.label_map[row['label']]
        
        # Collect frame paths and labels
        for video_folder in os.listdir(frames_root):
            video_path = os.path.join(frames_root, video_folder)
            if not os.path.isdir(video_path):
                continue
                
            # Get label from metadata
            label = video_to_label.get(video_folder)
            if label is None:
                continue
                
            # Collect all frames for this video
            for frame_file in os.listdir(video_path):
                if frame_file.endswith(('.jpg', '.jpeg', '.png')):
                    frame_path = os.path.join(video_path, frame_file)
                    self.samples.append((frame_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        frame_path, label = self.samples[idx]
        image = Image.open(frame_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

# ======================
# Data Transforms
# ======================
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# ======================
# Load Metadata
# ======================
metadata_df = pd.read_csv(METADATA_PATH)

# ======================
# Create Datasets and DataLoaders
# ======================
train_dataset = DeepFakeFrameDataset(
    TRAIN_FRAMES_ROOT,
    metadata_df,
    transform=train_transform
)

test_dataset = DeepFakeFrameDataset(
    TEST_FRAMES_ROOT,
    metadata_df,
    transform=test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# ======================
# Model Setup
# ======================
def create_model(num_classes=NUM_CLASSES):
    model = efficientnet_b4(pretrained=True)
    
    # Modify classifier head
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_features, num_classes))
    
    return model

model = create_model().to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max', 
    factor=0.5, 
    patience=2)

# ======================
# Training Loop
# ======================
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    best_f1 = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        
        # Validation phase
        val_loss, val_f1, _, _, _ = evaluate_model(model, test_loader, criterion)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {epoch_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val F1: {val_f1:.4f}')
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Best model saved with F1: {best_f1:.4f}')
        
        # Update learning rate
        scheduler.step(val_f1)
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load('best_model.pth'))
    return model, history

# ======================
# Evaluation Function
# ======================
def evaluate_model(model, test_loader, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    loss = running_loss / len(test_loader.dataset)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    return loss, f1, precision, recall, (all_labels, all_preds)

# ======================
# Plot Confusion Matrix
# ======================
def plot_confusion_matrix(true_labels, pred_labels, classes):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

# ======================
# Main Execution
# ======================
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"Train dataset size: {len(train_dataset)} frames")
    print(f"Test dataset size: {len(test_dataset)} frames")
    
    # Train the model
    trained_model, history = train_model(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        NUM_EPOCHS
    )
    
    # Final evaluation
    test_loss, test_f1, precision, recall, (true_labels, pred_labels) = evaluate_model(
        trained_model, test_loader, criterion
    )
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=['REAL', 'FAKE'], zero_division=0))
    
    print(f"\nTest F1 Score: {test_f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(true_labels, pred_labels, classes=['REAL', 'FAKE'])
    
    # Save predictions for analysis
    results_df = pd.DataFrame({
        'true_label': true_labels,
        'pred_label': pred_labels
    })
    results_df.to_csv('frame_predictions.csv', index=False)