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
from torch.cuda.amp import autocast

from train import UNet, SaltDataset

IMAGE_PIPELINE = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=45),
    transforms.Resize((224, 224), antialias=True),
    ])

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    destroy_process_group()

def train(rank, world_size):
    ddp_setup(rank, world_size)


    PARAMS = {

        'LOG_NAME' : "logs/Test4",
        'LR' : 0.001,
        'NUM_EPOCHS' : 2,
        'BATCH_SIZE' : 256,
        'SHUFFLE' : True,
        'NUM_WORKERS' : 4,
        'MIXED_PRECISION': True,
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

    # Create model and move it to the appropriate device
    model = UNet(in_channels=3, out_channels=1)
    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # Create data loader and sampler
    train_dataset = SaltDataset(root_dir='competition_data', mode='train', transform=IMAGE_PIPELINE)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=PARAMS['SHUFFLE'],)
    train_loader = DataLoader(train_dataset, batch_size=PARAMS['BATCH_SIZE'], num_workers=PARAMS['NUM_WORKERS'],
                              sampler=train_sampler)

    profile_on = True
    num_epochs = 2

    # Training loop
    for epoch in range(num_epochs): # Loop over the dataset multiple times
        now = time.time()
        ddp_model.train()
        train_sampler.set_epoch(epoch)

        # Iterate over the batches
        for inputs, masks in train_loader:

            optimizer.zero_grad()
            inputs = inputs.to(rank)
            masks = masks.to(rank)

            with autocast():
                outputs = ddp_model(inputs)
                loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()
            #running_loss += loss.item()

            if profile_on:
                profiler.step()

        if profile_on:
            profiler.stop()

        time_taken = time.time() - now
        print('Epoch [{}/{}]: {:2f}'.format(epoch+1, num_epochs, time_taken))

    cleanup()

    filename = os.listdir(PARAMS['LOG_NAME'])[0]
    #add_rank_key_to_log(os.path.join(PARAMS['LOG_NAME'], filename))

    PARAMS['TIME_TAKEN'] = time_taken

    # Save Params dict to log directory
    with open(os.path.join(PARAMS['LOG_NAME'], 'params.json'), 'w') as f:
        json.dump(PARAMS, f)

def main():

    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":

    main()


