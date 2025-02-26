import os
from monai.transforms import Resize, EnsureChannelFirst, Compose, NormalizeIntensity, ThresholdIntensity, AsDiscrete
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from torch.nn import BCELoss
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.nn.functional import sigmoid
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from weakly_supervised_library import CAL_WSISDataset
from collections import defaultdict
import pickle
import numpy as np


torch.manual_seed(4242)
n_epochs = 30
n_val_iter = 4
n_updates_per_epoch = None  #128  #None  # None means full epoch
opt_lr = 1e-4
log_every = 16

start_epoch = 7
load_from = f'fully_supervised_model_{start_epoch:02d}.pth' if start_epoch > 0 else None

# %%
path_to_aires_ganglionnaires = '/home/francescocremonesiext/new-areas/aires-ganglionnaires/output/cal_cancer_patients/SEG'
patient_list = os.listdir(path_to_aires_ganglionnaires)

# %%
to_remove = []
for i, pat in enumerate(patient_list):
    if len(os.listdir(os.path.join(path_to_aires_ganglionnaires, pat))) < 15:
        to_remove.append(pat)

# %%
for rm in to_remove:
    patient_list.pop(patient_list.index(rm))


# %%
common_shape = (128,128,64)
#common_shape = (32,32,64)
window=350
level=40
whole_dataset = CAL_WSISDataset(
    '/mnt/lvssd/common/AI4PET/data_v1.0.0/data/02_intermediate/CAL/CT',
    path_to_aires_ganglionnaires,
    patient_list,
    'level_iii_left',
    transforms = {
        'data': Compose((
                EnsureChannelFirst(channel_dim='no_channel'),
                Resize(common_shape),
                NormalizeIntensity(subtrahend=level-0.5*window, divisor=window),
                ThresholdIntensity(1, above=False, cval=1),
                ThresholdIntensity(0, above=True, cval=0),
                )),
        'label': Compose((
                EnsureChannelFirst(channel_dim='no_channel'),
                Resize(common_shape),
                AsDiscrete(threshold=0.5),
                )),
    },
    min_crop_shape=common_shape,  # deactivate crop_shape
    force_recompute_bbox=False,
    crop_images=True,
    only_crop_z=False
)

# %%
train_indices, test_indices = train_test_split(list(range(len(whole_dataset))), test_size=0.2, random_state=4242)

# %%
train_dataset = Subset(whole_dataset, train_indices)
test_dataset = Subset(whole_dataset, test_indices)


# %%
train_dataloader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
)

if n_updates_per_epoch is None:
    n_updates_per_epoch = len(train_dataloader)

# %%
if load_from is None:
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(32,64,128), 
        strides=(2,2),
        
    )
else:
    with open(load_from, 'rb') as f:
        model = torch.load(f, weights_only=False)


ce_loss_fn = BCELoss()
# for validation
dsc = DiceLoss()


# %%
optimizer = Adam(model.parameters(), lr=opt_lr)

# %%
for epoch in range(start_epoch+1,start_epoch+n_epochs+1):
    losses = defaultdict(lambda: [])

    val_iter = iter(test_dataloader)
    val_loss = 0.
    model.eval()
    n_val_iter_done = 0
    with torch.no_grad():
        for _ in tqdm(range(n_val_iter)):
            while(True):
                try:
                    img, lab = next(val_iter)
                    break
                except EOFError:
                    continue
            n_val_iter_done += 1
            pred = model(img)
            pred = sigmoid(pred)
            val_loss += dsc(pred,lab)
    print(f'Epoch {epoch:02d} Validation loss {val_loss/(1e-11+n_val_iter_done):0.8f}')

    model.train()
    data_iter = iter(train_dataloader)
    n_train_iter_done = 0
    train_loss = 0.
    for upd_it in tqdm(range(n_updates_per_epoch)):
        while(True):
            try:
                img, lab = next(data_iter)
                break
            except EOFError:
                continue
            except ValueError:
                continue
            except StopIteration:
                break

        optimizer.zero_grad()
        pred = model(img)
        pred = sigmoid(pred)

        ce_loss = ce_loss_fn(pred, lab)
        losses['ce'].append(ce_loss.detach().numpy())
        loss =  ce_loss 

        loss.backward()
        optimizer.step()
        n_train_iter_done += 1
        train_loss += loss.detach().numpy()

        if upd_it % log_every == log_every - 1:
            for lname, l in losses.items():
                lsum = np.array(l[-log_every:]).sum()
                print(f'Update {upd_it} ::  {lname}: {lsum/log_every:0.8f}')
            with open(f'fully_supervised_losses_{epoch:02d}.pkl', 'wb') as f:
                pickle.dump(dict(losses), f)
            
    print(f'Epoch {epoch:02d} Train loss {train_loss/(1e-11+n_train_iter_done):0.8f}')

    with open(f'fully_supervised_model_{epoch:02d}.pth', 'wb') as f:
        torch.save(model, f)


# %%
val_iter = iter(test_dataloader)
val_loss = 0.
model.eval()
n_val_iter_done = 0
with torch.no_grad():
    for _ in tqdm(range(n_val_iter)):
        while(True):
            try:
                img, lab = next(val_iter)
                break
            except EOFError:
                continue
            except ValueError:
                continue
        n_val_iter_done += 1
        pred = model(img)
        val_loss += dsc(pred,lab)
print(f'Final Validation loss {val_loss/(1e-11+n_val_iter_done):0.8f}')

