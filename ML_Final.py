import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from PIL import Image
import os
from tqdm import tqdm

class AgeInvariantDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.identities = []
        self.identity_to_images = {}
        
        if os.path.exists(data_root):
            for person_id in sorted(os.listdir(data_root)):
                person_path = os.path.join(data_root, person_id)
                if os.path.isdir(person_path):
                    images = [os.path.join(person_path, img) 
                             for img in os.listdir(person_path) 
                             if img.endswith(('.jpg', '.png', '.jpeg'))]
                    if len(images) >= 2:
                        self.identities.append(person_id)
                        self.identity_to_images[person_id] = images
        
        print(f"載入 {len(self.identities)} 個身份")
    
    def __len__(self):
        return len(self.identities)
    
    def __getitem__(self, idx):
        identity = self.identities[idx]
        images = self.identity_to_images[identity]
        
        loaded_images = []
        for img_path in images:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            loaded_images.append(img)
        
        return torch.stack(loaded_images), identity


def collate_triplet_batch(batch):
    images_list = []
    labels_list = []
    
    for identity_idx, (images, identity) in enumerate(batch):
        num_images = len(images)
        if num_images >= 4:
            selected_indices = torch.randperm(num_images)[:4]
        else:
            selected_indices = torch.arange(num_images)
        
        for idx in selected_indices:
            images_list.append(images[idx])
            labels_list.append(identity_idx)
    
    if len(images_list) == 0:
        return None, None
    
    return torch.stack(images_list), torch.tensor(labels_list)


class TripletNetwork(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=True):
        super(TripletNetwork, self).__init__()
        
        resnet = models.resnet50(pretrained=pretrained)
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        self.embedding = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_dim)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        embeddings = self.embedding(features)

        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class TripletLossHardMining(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLossHardMining, self).__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels):
        distance_matrix = torch.cdist(embeddings, embeddings, p=2).pow(2)

        triplet_loss = []
        
        for i in range(len(embeddings)):
            anchor_label = labels[i]

            positive_mask = (labels == anchor_label)
            positive_mask[i] = False
            
            if positive_mask.sum() == 0:
                continue
       
            positive_distances = distance_matrix[i][positive_mask]
            hardest_positive_dist = positive_distances.max()
            
            negative_mask = (labels != anchor_label)
            
            if negative_mask.sum() == 0:
                continue
            
            negative_distances = distance_matrix[i][negative_mask]
            hardest_negative_dist = negative_distances.min()
          
            loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
            triplet_loss.append(loss)
        
        if len(triplet_loss) == 0:
            return torch.tensor(0.0, requires_grad=True)
        
        return torch.stack(triplet_loss).mean()

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for images_batch, labels_batch in tqdm(dataloader, desc="Training"):
        if images_batch is None:
            continue
            
        images_batch = images_batch.to(device)
        labels_batch = labels_batch.to(device)
        
        embeddings = model(images_batch)
        loss = criterion(embeddings, labels_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    if num_batches == 0:
        return 0.0
    
    return total_loss / num_batches

def evaluate(model, test_dataset, device, num_pairs=1000):
    model.eval()
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            images, identity = test_dataset[idx]
            images = images.to(device)
            embeddings = model(images)
            
            all_embeddings.append(embeddings.cpu())
            all_labels.extend([identity] * len(images))
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    np.random.seed(42)
    distances = []
    labels = []
    
    for _ in range(num_pairs):
        idx1, idx2 = np.random.choice(len(all_embeddings), 2, replace=False)
        
        dist = torch.dist(all_embeddings[idx1], all_embeddings[idx2], p=2).item()
        distances.append(dist)
        
        label = 1 if all_labels[idx1] == all_labels[idx2] else 0
        labels.append(label)
    
    distances = np.array(distances)
    labels = np.array(labels)
    
    fpr, tpr, thresholds = roc_curve(labels, -distances)
    roc_auc = auc(fpr, tpr)
  
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    
    return fpr, tpr, roc_auc, eer, eer_threshold, distances, labels


def plot_roc_curve(fpr, tpr, roc_auc, eer):
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot([eer], [1-eer], 'ro', markersize=10, 
             label=f'EER = {eer:.4f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)', fontsize=12)
    plt.ylabel('True Positive Rate (TAR)', fontsize=12)
    plt.title('ROC Curve for Age-Invariant Face Recognition', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_distance_distribution(distances, labels):
    same_person = distances[labels == 1]
    diff_person = distances[labels == 0]
    
    plt.figure(figsize=(12, 6))
    plt.hist(same_person, bins=50, alpha=0.6, label='Same Person', color='green')
    plt.hist(diff_person, bins=50, alpha=0.6, label='Different Person', color='red')
    plt.xlabel('Euclidean Distance', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distance Distribution', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('distance_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    DATA_ROOT = '/content/face_dataset'
    
    EMBEDDING_DIM = 128
    MARGIN = 0.2
    P = 8  
    K = 4  
    BATCH_SIZE = P * K
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(DATA_ROOT):
        print(f"警告: 數據目錄 {DATA_ROOT} 不存在")
        print("請創建數據集或使用生成的模擬數據")

        os.makedirs(DATA_ROOT, exist_ok=True)
        for i in range(20): 
            person_dir = os.path.join(DATA_ROOT, f'person_{i:03d}')
            os.makedirs(person_dir, exist_ok=True)
            for j in range(5): 
                img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                img.save(os.path.join(person_dir, f'age_{j*10+5:02d}.jpg'))
        print(f"已創建模擬數據集於 {DATA_ROOT}")

    dataset = AgeInvariantDataset(DATA_ROOT, transform=transform)
    
    if len(dataset) == 0:
        print("錯誤: 數據集為空，請檢查數據路徑")
        return

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_identities = [dataset.identities[i] for i in train_indices]
    test_identities = [dataset.identities[i] for i in test_indices]
    
    train_dataset = AgeInvariantDataset.__new__(AgeInvariantDataset)
    train_dataset.data_root = dataset.data_root
    train_dataset.transform = dataset.transform
    train_dataset.identities = train_identities
    train_dataset.identity_to_images = {k: v for k, v in dataset.identity_to_images.items() if k in train_identities}
    
    test_dataset = AgeInvariantDataset.__new__(AgeInvariantDataset)
    test_dataset.data_root = dataset.data_root
    test_dataset.transform = dataset.transform
    test_dataset.identities = test_identities
    test_dataset.identity_to_images = {k: v for k, v in dataset.identity_to_images.items() if k in test_identities}
    
    print(f"訓練集: {len(train_dataset)} 個身份")
    print(f"測試集: {len(test_dataset)} 個身份")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=P, 
        shuffle=True,
        collate_fn=collate_triplet_batch,
        num_workers=0
    )
    
    model = TripletNetwork(embedding_dim=EMBEDDING_DIM, pretrained=True)
    model = model.to(device)
    
    criterion = TripletLossHardMining(margin=MARGIN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    print("\n開始訓練...")
    train_losses = []
    
    for epoch in range(NUM_EPOCHS):
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(avg_loss)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print("評估模型...")
            fpr, tpr, roc_auc, eer, eer_threshold, distances, labels = evaluate(
                model, test_dataset, device
            )
            print(f"ROC AUC: {roc_auc:.4f}, EER: {eer:.4f}, EER Threshold: {eer_threshold:.4f}")
    
    print("\n最終評估...")
    fpr, tpr, roc_auc, eer, eer_threshold, distances, labels = evaluate(
        model, test_dataset, device, num_pairs=2000
    )
    
    print(f"\n最終結果:")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"EER: {eer:.4f}")
    print(f"EER Threshold: {eer_threshold:.4f}")
    
    plot_roc_curve(fpr, tpr, roc_auc, eer)
    plot_distance_distribution(distances, labels)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_EPOCHS+1), train_losses, marker='o')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    torch.save(model.state_dict(), 'age_invariant_model.pth')
    print("\n模型已保存至 age_invariant_model.pth")


if __name__ == "__main__":
    main()
