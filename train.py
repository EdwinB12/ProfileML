import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.profiler import schedule
#from hta.trace_analysis import TraceAnalysis
import os
import json
import time
from PIL import Image
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.cuda.amp import GradScaler, autocast

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, multiplier=1):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64 * multiplier, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64* multiplier, 64* multiplier, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64* multiplier, 128* multiplier, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128* multiplier, 128* multiplier, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128* multiplier, 64* multiplier, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64* multiplier, 64* multiplier, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64 * multiplier, out_channels, kernel_size=2, stride=2),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        #print(x2.shape)
        x3 = self.decoder(x2)
        #print(x3.shape)
        return x3

class SaltDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = os.path.join(root_dir, mode)
        self.mode = mode
        self.transform = transform
        self.image_ids = os.listdir(os.path.join(self.root_dir, 'images'))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.root_dir, 'images', image_id)
        mask_path = os.path.join(self.root_dir, 'masks', image_id)

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

IMAGE_PIPELINE = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=45),
    transforms.Resize((224, 224), antialias=True),
    ])

def add_rank_key_to_log(log_path):

    # Load the json file, check if "distributedInfo" key exists,
    # if not, then add it with the value "distributedInfo": {"rank": 0}

    # Load the json file
    with open(log_path, 'r') as f:
        log = json.load(f)

    # Check if "distributedInfo" key exists
    if "distributedInfo" not in log:
        log["distributedInfo"] = {"rank": 0}

    # Write the updated log back to the file
    with open(log_path, 'w') as f:
        json.dump(log, f)

def train(model, num_epochs, train_loader, criterion, optimizer, profile_on_list, device, profiler, mixed_precision):

    # Iterate over Epochs
    for epoch in range(num_epochs):
        now = time.time()

        profile_on = profile_on_list[epoch]

        running_loss = 0.0

        print(f"Epoch: {epoch}")

        model.train()

        if profile_on:
            profiler.start()

        # Iterate over the batches
        for inputs, masks in train_loader:

            optimizer.zero_grad()
            inputs = inputs.to(device)
            masks = masks.to(device)

            if mixed_precision:
                with autocast():

                    outputs = model(inputs)
                    loss = criterion(outputs, masks)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if profile_on:
                profiler.step()

        if profile_on:
            profiler.stop()

        time_taken = time.time() - now
        print('Epoch [{}/{}]: {:2f}'.format(epoch+1, num_epochs, time_taken))

    return time_taken

if __name__ == "__main__":

    PARAMS = {

        'LOG_NAME' : "logs/Test1",
        'LR' : 0.001,
        'NUM_EPOCHS' : 2,
        'BATCH_SIZE' : 256,
        'SHUFFLE' : True,
        'NUM_WORKERS' : 0,
        'MIXED_PRECISION': False,
    }

    my_schedule = schedule(
            skip_first=5,
            wait=2,
            warmup=2,
            active=5,
            repeat=1)

    profiler = torch.profiler.profile(
                schedule=my_schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(PARAMS['LOG_NAME']),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )

    device = 'cuda'

    train_dataset = SaltDataset(root_dir='competition_data', mode='train', transform=IMAGE_PIPELINE)
    # Test 1
    train_loader = DataLoader(train_dataset, batch_size=PARAMS['BATCH_SIZE'], shuffle=PARAMS['SHUFFLE'], num_workers=PARAMS['NUM_WORKERS'])

    model = UNet(in_channels=3, out_channels=1)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=PARAMS['LR'])

    time_taken = train(model, PARAMS['NUM_EPOCHS'], train_loader, criterion, optimizer,
                            profile_on_list=[True, False], device=device, profiler=profiler,
                            mixed_precision=PARAMS['MIXED_PRECISION'])

    filename = os.listdir(PARAMS['LOG_NAME'])[0]
    add_rank_key_to_log(os.path.join(PARAMS['LOG_NAME'], filename))

    PARAMS['TIME_TAKEN'] = time_taken

    # Save Params dict to log directory
    with open(os.path.join(PARAMS['LOG_NAME'], 'params.json'), 'w') as f:
        json.dump(PARAMS, f)

