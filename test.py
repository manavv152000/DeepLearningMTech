import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # fixed output size
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def vgg(cfg, num_classes=10, batch_norm=True):
    return VGG(make_layers(cfg, batch_norm=batch_norm), num_classes=num_classes)

def get_test_data(batch_size=64):
    """Load CIFAR-10 test dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    print("Loading CIFAR-10 test dataset...")
    test_data = datasets.CIFAR10('./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Test dataset loaded: {len(test_data)} samples")
    return test_loader

def evaluate_model(model, test_loader, device):
    """Evaluate model on test dataset"""
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    # CIFAR-10 class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("Evaluating model on test dataset...")
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Per-class accuracy
            for i in range(target.size(0)):
                label = target[i]
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
            
            if batch_idx % 50 == 0:
                print(f"Processed {batch_idx * len(data)}/{len(test_loader.dataset)} samples")
    
    # Overall accuracy
    overall_accuracy = 100. * correct / total
    
    # Per-class accuracy
    print("\n" + "="*60)
    print(f"OVERALL TEST ACCURACY: {overall_accuracy:.2f}%")
    print("="*60)
    print("\nPer-class Accuracy:")
    print("-" * 30)
    for i in range(10):
        if class_total[i] > 0:
            class_acc = 100. * class_correct[i] / class_total[i]
            print(f"{classes[i]:12s}: {class_acc:6.2f}% ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"{classes[i]:12s}: No samples")
    print("-" * 30)
    
    return overall_accuracy

def load_trained_model(model_path='trained_vgg6_model.pth'):
    """Load the trained model from checkpoint"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Get model configuration
        cfg_vgg6 = checkpoint.get('model_config', [64, 64, 'M', 128, 128, 'M'])
        
        # Create model with same architecture
        model = vgg(cfg_vgg6, num_classes=10, batch_norm=True)
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Model was trained for {checkpoint.get('epochs', 'unknown')} epochs")
        
        return model
    
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found!")
        print("Please make sure you have trained and saved the model first.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the trained model
    model = load_trained_model()
    if model is None:
        exit(1)
    
    model = model.to(device)
    
    # Load test data
    test_loader = get_test_data(batch_size=64)
    
    # Evaluate the model
    test_accuracy = evaluate_model(model, test_loader, device)
    
    print(f"\nFinal Result: Test Accuracy = {test_accuracy:.2f}%")