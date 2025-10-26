import torch
from models_cleaned import VGG6
from train_utils import get_dataloaders, evaluate
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = VGG6('relu').to(device)
# Please edit the path accordingly
model.load_state_dict(torch.load('C:/Users/saiki/Downloads/Assignment/best_model.pth', map_location=device))
model.eval()

# Evaluate on CIFAR-10 test data
criterion = nn.CrossEntropyLoss()
_, _, test_loader = get_dataloaders(batch_size=128)
test_loss, test_acc = evaluate(model, device, test_loader, criterion)

print(" Model tested successfully. Test Accuracy:",test_acc)