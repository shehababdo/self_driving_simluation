import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
import numpy as np
import os
import pandas as pd
import gc
import matplotlib.pyplot as plt

gc.collect()

torch.cuda.empty_cache()

class LoadData(Dataset):
    def __init__(self, label_path, img_dir):
        self.img_labels = pd.read_csv(label_path)
        self.img_dir = img_dir    

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path_center = os.path.join(self.img_dir, self.img_labels.iloc[idx]['Center Image'])
        steering_angle = self.img_labels.iloc[idx]['Steering']
        
        self.transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToPILImage(),
        transforms.ToTensor()
                                            ])
        
        image_center = read_image(img_path_center)
        image_center = self.transform(image_center)

        steering_angle = torch.tensor(steering_angle, dtype=torch.float32)
    
        return {
            'Center Image': image_center,
            'Steering': steering_angle,
        }
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

batch_size =16
learning_rate = 0.001
num_epochs = 10



training_path = r'C:\Users\Shehab Abdo\Documents\data\driving_log.csv'
validation_path = r'C:\Users\Shehab Abdo\Documents\validation_data\driving_log.csv'
training_path_IMG = r'C:\Users\Shehab Abdo\Documents\data\IMG'
validation_path_IMG = r'C:\Users\Shehab Abdo\Documents\validation_data\IMG'

train_dataset = LoadData(training_path, training_path_IMG)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = LoadData(validation_path, validation_path_IMG)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(25600, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

model = MyModel().to(device)

criterion_steering_angle_loss_fn=nn.MSELoss()

optimizer_steering_angle_loss_fn=optim.Adam(model.parameters(), lr=learning_rate)


print("Training Has Just Started")
training_losses = []
for epoch in range(num_epochs):
    for i,data in enumerate(train_loader):
        
      images_center=data['Center Image'].to(device) 
      steering_angles=data['Steering'].to(device) 

      outputs=model(images_center) 
      
      loss_steering_angle=criterion_steering_angle_loss_fn(outputs[:,0],steering_angles) 
      
      optimizer_steering_angle_loss_fn.zero_grad() 
      loss_steering_angle.backward() 
      optimizer_steering_angle_loss_fn.step() 
      training_losses.append(loss_steering_angle.item())
      if (i+1)%12==0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Training Loss: {loss_steering_angle.item():.4f}")
    
print("Training Has Just Finished")



model.eval()
val_losses=[]
with torch.no_grad():
    print("Validation Has Just Started")
    total_loss_steering_angle=0
    total_samples=0

    for data in val_loader:
        images_center=data['Center Image'].to(device) 
        steering_angles=data['Steering'].to(device) 

        outputs=model(images_center) 

        loss_steering_angle=criterion_steering_angle_loss_fn(outputs[:,0],steering_angles) 

        total_loss_steering_angle+=loss_steering_angle.item()*steering_angles.size(0)
        total_samples+=steering_angles.size(0)
      #  print(f"Validation Loss: {loss_steering_angle.item():.4f}")
        val_losses.append(loss_steering_angle.item())
    avg_loss_steering_angle=total_loss_steering_angle/total_samples
    print("Validation Has Just Finished")
    print(f"Validation Losses: {avg_loss_steering_angle:.4f}")



# Save your model
model_dir = r'C:\Users\Shehab Abdo\Documents\model path'
os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist

model_path = os.path.join(model_dir, 'model.pt')
torch.save(model.state_dict(), model_path)
print("Model saved at", model_path)

plt.figure(figsize=(10,5))
plt.title("Training Loss VS Val")
plt.plot(training_losses)
plt.plot(val_losses)
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.show()