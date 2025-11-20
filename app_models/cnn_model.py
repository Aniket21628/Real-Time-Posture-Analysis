"""
Custom CNN Model for Sitting Posture Detection
This module contains a custom Convolutional Neural Network implementation
for binary classification of sitting postures (good vs bad).

Note: This is an alternative implementation to the YOLOv5 approach.
The current production system uses YOLOv5 for better real-time performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class PostureCNN(nn.Module):
    """
    Custom CNN architecture for sitting posture classification.
    
    Architecture:
    - 3 Convolutional blocks with batch normalization and max pooling
    - 2 Fully connected layers with dropout for regularization
    - Binary classification output (good/bad posture)
    
    Input: RGB images of shape (3, 224, 224)
    Output: 2 class probabilities (sitting_good, sitting_bad)
    """
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(PostureCNN, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 4
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the flattened size after convolutions
        # Input: 224x224 -> after 4 pooling layers: 14x14
        self.flatten_size = 256 * 14 * 14
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Convolutional Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Convolutional Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Convolutional Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Convolutional Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        
        # Flatten
        x = x.view(-1, self.flatten_size)
        
        # Fully Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x
    
    def predict(self, x):
        """
        Make predictions with softmax probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (predicted_class, probabilities)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        return predicted_class, probabilities


class PostureCNNDeep(nn.Module):
    """
    Deeper CNN architecture with residual-like connections for improved accuracy.
    
    This architecture includes:
    - 5 Convolutional blocks with increasing depth
    - Batch normalization for training stability
    - Dropout for regularization
    - Suitable for more complex posture detection scenarios
    """
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(PostureCNNDeep, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: 112x112 -> 56x56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: 56x56 -> 28x28
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4: 28x28 -> 14x14
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512 * 14 * 14, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CNNInferenceModel:
    """
    Wrapper class for CNN model inference, compatible with the existing codebase.
    This provides a similar interface to the YOLOv5 InferenceModel class.
    """
    
    def __init__(self, model_path, device='cpu', model_type='standard'):
        """
        Initialize the CNN inference model.
        
        Args:
            model_path: Path to the saved model weights (.pth file)
            device: Device to run inference on ('cpu' or 'cuda')
            model_type: Type of model ('standard' or 'deep')
        """
        self.device = torch.device(device)
        self.class_names = ['sitting_good', 'sitting_bad']
        
        # Initialize model architecture
        if model_type == 'deep':
            self.model = PostureCNNDeep(num_classes=2)
        else:
            self.model = PostureCNN(num_classes=2)
        
        # Load trained weights if path exists
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model weights from {model_path}")
        except FileNotFoundError:
            print(f"Warning: Model weights not found at {model_path}")
            print("Using randomly initialized weights (model not trained)")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def predict(self, image):
        """
        Perform inference on an input image.
        
        Args:
            image: Input image (numpy array in BGR format from OpenCV)
            
        Returns:
            Dictionary containing prediction results
        """
        # Preprocess image
        import cv2
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        # Format results
        class_name = self.class_names[predicted_class.item()]
        confidence_value = confidence.item()
        
        return {
            'class': class_name,
            'confidence': confidence_value,
            'class_id': predicted_class.item(),
            'probabilities': probabilities.cpu().numpy()[0]
        }


def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, device='cpu'):
    """
    Training function for the CNN model.
    
    Args:
        model: CNN model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on
        
    Returns:
        Trained model
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                           factor=0.5, patience=5)
    
    best_val_accuracy = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_accuracy = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_posture_cnn.pth')
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        print('-' * 60)
    
    return model


if __name__ == "__main__":
    # Example usage and model summary
    print("=" * 60)
    print("Custom CNN Model for Sitting Posture Detection")
    print("=" * 60)
    
    # Initialize standard model
    model_standard = PostureCNN(num_classes=2)
    print("\n[Standard CNN Architecture]")
    print(f"Total parameters: {sum(p.numel() for p in model_standard.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model_standard.parameters() if p.requires_grad):,}")
    
    # Initialize deep model
    model_deep = PostureCNNDeep(num_classes=2)
    print("\n[Deep CNN Architecture]")
    print(f"Total parameters: {sum(p.numel() for p in model_deep.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model_deep.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model_standard(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("\nModel initialized successfully!")
