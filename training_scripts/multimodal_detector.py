import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import librosa
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")

# ======================
# Configuration
# ======================
CONFIG = {
    "image": {
        "model_name": "efficientnet_b0",
        "input_size": (224, 224),
        "batch_size": 32,
        "lr": 1e-4,
        "epochs": 1
    },
    "audio": {
        "model_name": "ecapa_tdnn",
        "sample_rate": 16000,
        "duration": 5,  # seconds
        "batch_size": 16,
        "lr": 1e-4,
        "epochs": 5,
        "local_model_path": "./pretrained_models/ecapa_tdnn"  # Local path for offline use
    },
    "text": {
        "model_name": "xlm-roberta-base",
        "max_length": 128,  
        "batch_size": 16,
        "lr": 2e-5,
        "epochs": 5,
        "local_model_path": "./pretrained_models/xlm-roberta-base"  # Local path for offline use
    },
    "video": {
        "model_name": "r3d_18",
        "clip_length": 16,  # number of frames
        "frame_size": (112, 112),
        "batch_size": 8,
        "lr": 1e-4,
        "epochs": 5
    }
}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
METADATA_PATH = "../notebooks/data/HAV_DF/video_metadata.csv"

# Create directories for local models
os.makedirs(CONFIG["audio"]["local_model_path"], exist_ok=True)
os.makedirs(CONFIG["text"]["local_model_path"], exist_ok=True)

# ======================
# Dataset Classes
# ======================
class ImageDataset(Dataset):
    def __init__(self, frames_root, metadata_df, transform=None):
        self.samples = []
        self.transform = transform
        self.label_map = {'REAL': 0, 'FAKE': 1}
        
        video_to_label = {}
        for _, row in metadata_df.iterrows():
            video_name = os.path.splitext(row['video_name'])[0]
            video_to_label[video_name] = self.label_map[row['label']]
        
        for video_folder in os.listdir(frames_root):
            video_path = os.path.join(frames_root, video_folder)
            if not os.path.isdir(video_path):
                continue
                
            label = video_to_label.get(video_folder)
            if label is None:
                continue
                
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


class AudioDataset(Dataset):
    def __init__(self, audio_root, metadata_df, sample_rate=16000, duration=5):
        self.samples = []
        self.sample_rate = sample_rate
        self.duration = duration
        self.label_map = {'REAL': 0, 'FAKE': 1}
        
        video_to_label = {}
        for _, row in metadata_df.iterrows():
            video_name = os.path.splitext(row['video_name'])[0]
            video_to_label[video_name] = self.label_map[row['label']]
        
        for audio_file in os.listdir(audio_root):
            if audio_file.endswith('.wav'):
                audio_path = os.path.join(audio_root, audio_file)
                video_name = os.path.splitext(audio_file)[0]
                label = video_to_label.get(video_name)
                if label is not None:
                    self.samples.append((audio_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        
        try:
            waveform, _ = librosa.load(audio_path, sr=self.sample_rate, 
                                      duration=self.duration, mono=True)
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            waveform = np.zeros(int(self.sample_rate * self.duration))
        
        # Pad or trim to fixed length
        target_length = self.sample_rate * self.duration
        if len(waveform) < target_length:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))
        else:
            waveform = waveform[:target_length]
            
        return torch.tensor(waveform, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class TextDataset(Dataset):
    def __init__(self, text_root, metadata_df, tokenizer, max_length=128):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {'REAL': 0, 'FAKE': 1}
        
        video_to_label = {}
        for _, row in metadata_df.iterrows():
            video_name = os.path.splitext(row['video_name'])[0]
            video_to_label[video_name] = self.label_map[row['label']]
        
        for text_file in os.listdir(text_root):
            if text_file.endswith('.txt'):
                text_path = os.path.join(text_root, text_file)
                video_name = os.path.splitext(text_file)[0]
                label = video_to_label.get(video_name)
                if label is not None:
                    try:
                        with open(text_path, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                        self.samples.append((text, label))
                    except Exception as e:
                        print(f"Error loading text {text_path}: {e}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }, torch.tensor(label, dtype=torch.long)


class VideoDataset(Dataset):
    def __init__(self, frames_root, metadata_df, clip_length=16, transform=None):
        self.samples = []
        self.clip_length = clip_length
        self.transform = transform
        self.label_map = {'REAL': 0, 'FAKE': 1}
        
        video_to_label = {}
        for _, row in metadata_df.iterrows():
            video_name = os.path.splitext(row['video_name'])[0]
            video_to_label[video_name] = self.label_map[row['label']]
        
        for video_folder in os.listdir(frames_root):
            video_path = os.path.join(frames_root, video_folder)
            if not os.path.isdir(video_path):
                continue
                
            label = video_to_label.get(video_folder)
            if label is None:
                continue
                
            frame_files = sorted([
                f for f in os.listdir(video_path) 
                if f.endswith(('.jpg', '.jpeg', '.png'))
            ])
            
            # Skip videos with insufficient frames
            if len(frame_files) < clip_length:
                continue
                
            self.samples.append((video_path, frame_files, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, frame_files, label = self.samples[idx]
        
        # Select equally spaced frames
        total_frames = len(frame_files)
        indices = np.linspace(0, total_frames - 1, self.clip_length, dtype=int)
        selected_frames = [frame_files[i] for i in indices]
        
        frames = []
        for frame_file in selected_frames:
            frame_path = os.path.join(video_path, frame_file)
            try:
                frame = Image.open(frame_path).convert('RGB')
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            except Exception as e:
                print(f"Error loading frame {frame_path}: {e}")
                # Use a blank frame as fallback
                blank_frame = torch.zeros(3, self.transform.transforms[0].size[0], 
                                         self.transform.transforms[0].size[1])
                frames.append(blank_frame)
        
        # Stack frames to form a clip (C, T, H, W)
        clip = torch.stack(frames, dim=1)
        return clip, torch.tensor(label, dtype=torch.long)


# ======================
# Model Definitions with Offline Support
# ======================
class ImageModel(nn.Module):
    def __init__(self, model_name='efficientnet_b0', num_classes=2):
        super().__init__()
        if model_name == 'efficientnet_b0':
            self.model = torchvision.models.efficientnet_b0(pretrained=True)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x):
        return self.model(x)


class AudioModel(nn.Module):
    def __init__(self, model_name='ecapa_tdnn', num_classes=2):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = None
        self.load_weights()  # Initialize model during construction
    
    def load_weights(self):
        """Load weights separately to handle offline case"""
        if self.model_name == 'ecapa_tdnn':
            try:
                # Try to import with offline support
                from speechbrain.inference import EncoderClassifier
                
                # Check if local model exists
                if os.path.exists(CONFIG["audio"]["local_model_path"]) and \
                   os.path.exists(os.path.join(CONFIG["audio"]["local_model_path"], "hyperparams.yaml")):
                    print("Loading ECAPA-TDNN from local cache")
                    self.model = EncoderClassifier.from_hparams(
                        source=CONFIG["audio"]["local_model_path"],
                        savedir=CONFIG["audio"]["local_model_path"]
                    )
                    
                    # Freeze feature extractor
                    for param in self.model.parameters():
                        param.requires_grad = False
                    
                    # Replace classifier if available
                    if hasattr(self.model, 'modules') and hasattr(self.model.modules, 'classifier'):
                        in_features = self.model.modules['classifier'].in_features
                        self.model.modules['classifier'] = nn.Linear(in_features, self.num_classes)
                else:
                    print("Downloading ECAPA-TDNN model")
                    self.model = EncoderClassifier.from_hparams(
                        source="speechbrain/spkrec-ecapa-voxceleb",
                        savedir=CONFIG["audio"]["local_model_path"]
                    )
                    
                    # Freeze feature extractor
                    for param in self.model.parameters():
                        param.requires_grad = False
                    
                    # Replace classifier
                    in_features = self.model.modules['classifier'].in_features
                    self.model.modules['classifier'] = nn.Linear(in_features, self.num_classes)
            except Exception as e:
                print(f"Error loading audio model: {e}")
                print("Using fallback audio model (MLP)")
                self.model = nn.Sequential(
                    nn.Linear(16000 * 5, 512),
                    nn.ReLU(),
                    nn.Linear(512, self.num_classes)
                )
        else:
            raise ValueError(f"Unsupported audio model: {self.model_name}")
    
    def forward(self, x):
        # Handle different input types
        if isinstance(self.model, nn.Module):
            # For PyTorch models
            return self.model(x)
        else:
            # For SpeechBrain models
            return self.model(x.unsqueeze(1))

class TextModel(nn.Module):
    def __init__(self, model_name='xlm-roberta-base', num_classes=2):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = None
        self.load_weights()  # Initialize model during construction
        
        # Ensure model is properly initialized
        if self.model is None:
            print("Critical error: Text model not initialized. Using simple linear model.")
            self.model = nn.Linear(128, num_classes)
    
    def load_weights(self):
        """Load weights separately to handle offline case"""
        try:
            from transformers import XLMRobertaForSequenceClassification
            
            # Check if local model exists
            if os.path.exists(CONFIG["text"]["local_model_path"]) and \
               os.path.exists(os.path.join(CONFIG["text"]["local_model_path"], "config.json")):
                print("Loading XLM-R from local cache")
                self.model = XLMRobertaForSequenceClassification.from_pretrained(
                    CONFIG["text"]["local_model_path"],
                    num_labels=self.num_classes
                )
            else:
                print("Downloading XLM-R model")
                self.model = XLMRobertaForSequenceClassification.from_pretrained(
                    "xlm-roberta-base",
                    num_labels=self.num_classes
                )
                # Save model for offline use
                self.model.save_pretrained(CONFIG["text"]["local_model_path"])
        except Exception as e:
            print(f"Error loading text model: {e}")
            print("Using fallback text model (Linear)")
            self.model = nn.Linear(128, self.num_classes)
    
    def forward(self, input_ids, attention_mask):
        if hasattr(self.model, 'config'):  # Hugging Face model
            return self.model(input_ids, attention_mask=attention_mask)
        else:
            # Fallback model - use only the first token for simplicity
            return self.model(input_ids[:, 0].float())


class VideoModel(nn.Module):
    def __init__(self, model_name='r3d_18', num_classes=2):
        super().__init__()
        if model_name == 'r3d_18':
            self.model = torchvision.models.video.r3d_18(pretrained=True)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x):
        return self.model(x)


# ======================
# Training and Evaluation
# ======================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device):
    best_val_auc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        for data, labels in progress_bar:
            # Handle different input types
            if isinstance(data, dict):  # For text inputs
                inputs = {k: v.to(device) for k, v in data.items()}
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(**inputs)
                loss = criterion(outputs.logits, labels) if hasattr(outputs, 'logits') else criterion(outputs, labels)
                batch_size = data['input_ids'].size(0)  # Get batch size from input_ids
            else:
                inputs = data.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                batch_size = inputs.size(0)  # Get batch size from inputs
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_size
            progress_bar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # Validation
        val_loss, val_auc, _, _ = evaluate_model(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        print(f'Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val AUC: {val_auc:.4f}')
        
        # Update scheduler
        if scheduler:
            scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), f'best_{model.__class__.__name__}.pth')
            print(f'Best model saved with AUC: {best_val_auc:.4f}')
    
    return history


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    running_loss = 0.0
    
    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc='Evaluating'):
            if isinstance(data, dict):  # For text inputs
                inputs = {k: v.to(device) for k, v in data.items()}
                labels = labels.to(device)
                outputs = model(**inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                batch_size = data['input_ids'].size(0)  # Get batch size from input_ids
            else:
                inputs = data.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                logits = outputs
                batch_size = inputs.size(0)  # Get batch size from inputs
            
            loss = criterion(logits, labels)
            running_loss += loss.item() * batch_size
            
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)
            
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of FAKE class
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    loss = running_loss / len(test_loader.dataset)
    
    try:
        auc = roc_auc_score(all_labels, all_probs  )
    except ValueError:
        auc = 0.5  # Default to random chance if AUC can't be calculated
    
    return loss, auc, all_labels, all_preds


def print_metrics(labels, preds, model_name):
    print(f"\n{model_name} Evaluation Metrics:")
    print(classification_report(labels, preds, target_names=['REAL', 'FAKE'], zero_division=0))
    
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['REAL', 'FAKE'], 
                yticklabels=['REAL', 'FAKE'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()  # Close plot to prevent display in CLI
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


# ======================
# Main Execution
# ======================
def main():
    # Load metadata
    metadata_df = pd.read_csv(METADATA_PATH)
    
    # ======================
    # Image Modality (EfficientNet-B0)
    # ======================
    # print("\n=== Training Image Modality (EfficientNet-B0) ===")
    img_config = CONFIG['image']
    img_transform = transforms.Compose([
        transforms.Resize(img_config['input_size']),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # # Create datasets
    train_img_dataset = ImageDataset(
        "../notebooks/data/processed_data/train/frames",
        metadata_df,
        transform=img_transform
    )
    test_img_dataset = ImageDataset(
        "../notebooks/data/processed_data/test/frames",
        metadata_df,
        transform=img_transform
    )
    
    # # # Create dataloaders
    train_img_loader = DataLoader(
        train_img_dataset,
        batch_size=img_config['batch_size'],
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    test_img_loader = DataLoader(
        test_img_dataset,
        batch_size=img_config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # # Initialize model
    img_model = ImageModel(model_name=img_config['model_name']).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(img_model.parameters(), lr=img_config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # # Train and evaluate
    train_model(
        img_model, train_img_loader, test_img_loader, 
        criterion, optimizer, scheduler, img_config['epochs'], DEVICE
    )
    _, img_auc, img_labels, img_preds = evaluate_model(
        img_model, test_img_loader, criterion, DEVICE
    )
    print_metrics(img_labels, img_preds, "Image Modality")
    
    # ======================
    # Audio Modality (ECAPA-TDNN)
    # ======================
    print("\n=== Training Audio Modality (ECAPA-TDNN) ===")
    audio_config = CONFIG['audio']
    
    # Create datasets
    train_audio_dataset = AudioDataset(
        "../notebooks/data/processed_data/train/audio",
        metadata_df,
        sample_rate=audio_config['sample_rate'],
        duration=audio_config['duration']
    )
    test_audio_dataset = AudioDataset(
        "../notebooks/data/processed_data/test/audio",
        metadata_df,
        sample_rate=audio_config['sample_rate'],
        duration=audio_config['duration']
    )
    
    # Create dataloaders
    train_audio_loader = DataLoader(
        train_audio_dataset,
        batch_size=audio_config['batch_size'],
        shuffle=True
    )
    test_audio_loader = DataLoader(
        test_audio_dataset,
        batch_size=audio_config['batch_size'],
        shuffle=False
    )
    
    # Initialize model
    audio_model = AudioModel(model_name=audio_config['model_name']).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(audio_model.parameters(), lr=audio_config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # Train and evaluate
    train_model(
        audio_model, train_audio_loader, test_audio_loader, 
        criterion, optimizer, scheduler, audio_config['epochs'], DEVICE
    )
    _, audio_auc, audio_labels, audio_preds = evaluate_model(
        audio_model, test_audio_loader, criterion, DEVICE
    )
    print_metrics(audio_labels, audio_preds, "Audio Modality")
    
    # ======================
    # Text Modality (XLM-R)
    # ======================
    print("\n=== Training Text Modality (XLM-R) ===")
    text_config = CONFIG['text']
    tokenizer = XLMRobertaTokenizer.from_pretrained(text_config['model_name'])
    
    # Create datasets
    train_text_dataset = TextDataset(
        "../notebooks/data/processed_data/train/transcripts",
        metadata_df,
        tokenizer,
        max_length=text_config['max_length']
    )
    test_text_dataset = TextDataset(
        "../notebooks/data/processed_data/test/transcripts",
        metadata_df,
        tokenizer,
        max_length=text_config['max_length']
    )
    
    # Create dataloaders
    train_text_loader = DataLoader(
        train_text_dataset,
        batch_size=text_config['batch_size'],
        shuffle=True
    )
    test_text_loader = DataLoader(
        test_text_dataset,
        batch_size=text_config['batch_size'],
        shuffle=False
    )
    
    # Initialize model
    text_model = TextModel(model_name=text_config['model_name']).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(text_model.parameters(), lr=text_config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=1
    )
    
    # Train and evaluate
    train_model(
        text_model, train_text_loader, test_text_loader, 
        criterion, optimizer, scheduler, text_config['epochs'], DEVICE
    )
    _, text_auc, text_labels, text_preds = evaluate_model(
        text_model, test_text_loader, criterion, DEVICE
    )
    print_metrics(text_labels, text_preds, "Text Modality")
    
    # ======================
    # Video Modality (3D ResNet-18)
    # ======================
    print("\n=== Training Video Modality (3D ResNet-18) ===")
    video_config = CONFIG['video']
    video_transform = transforms.Compose([
        transforms.Resize(video_config['frame_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], 
                             std=[0.22803, 0.22145, 0.216989])
    ])
    
    # Create datasets
    train_video_dataset = VideoDataset(
        "../notebooks/data/processed_data/train/frames",
        metadata_df,
        clip_length=video_config['clip_length'],
        transform=video_transform
    )
    test_video_dataset = VideoDataset(
        "../notebooks/data/processed_data/test/frames",
        metadata_df,
        clip_length=video_config['clip_length'],
        transform=video_transform
    )
    
    # Create dataloaders
    train_video_loader = DataLoader(
        train_video_dataset,
        batch_size=video_config['batch_size'],
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    test_video_loader = DataLoader(
        test_video_dataset,
        batch_size=video_config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    video_model = VideoModel(model_name=video_config['model_name']).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(video_model.parameters(), lr=video_config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # Train and evaluate
    train_model(
        video_model, train_video_loader, test_video_loader, 
        criterion, optimizer, scheduler, video_config['epochs'], DEVICE
    )
    _, video_auc, video_labels, video_preds = evaluate_model(
        video_model, test_video_loader, criterion, DEVICE
    )
    print_metrics(video_labels, video_preds, "Video Modality")
    
    # ======================
    # Final Comparison
    # ======================
    print("\n=== Final Results ===")
    print(f"Image Modality (EfficientNet-B0) AUC: {img_auc:.4f}")
    print(f"Audio Modality (ECAPA-TDNN) AUC: {audio_auc:.4f}")
    print(f"Text Modality (XLM-R) AUC: {text_auc:.4f}")
    print(f"Video Modality (3D ResNet-18) AUC: {video_auc:.4f}")
    
    # Plot comparison
    modalities = ['Image', 'Audio', 'Text', 'Video'] #'Image',
    auc_scores = [img_auc, audio_auc, text_auc, video_auc] #img_auc
    
    plt.figure(figsize=(10, 6))
    plt.bar(modalities, auc_scores, color=['blue', 'orange', 'green', 'red'])
    plt.ylim(0, 1.0)
    plt.ylabel('AUC Score')
    plt.title('Deepfake Detection Performance by Modality')
    for i, v in enumerate(auc_scores):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
    plt.savefig('modality_comparison.png')
    plt.close()  # Close plot to prevent display in CLI


if __name__ == "__main__":
    main()