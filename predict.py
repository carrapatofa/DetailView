# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 09:21:58 2023

@author: Julian
"""

# import packages
import os
import torch
import numpy as np
import pandas as pd
from torchvision import transforms

# import own scripts
import parallel_densenet as net

# set parameters
n_class = 33    # number of classes
n_view  = 7     # number of views
n_batch = 2**1  # batch size (evtl hochschrauben? mehr memory?)
n_train = 2**13 # training dataset size
res = 256       # image ressolution
n_sides = n_view - 3      # number of sideviews

# set paths
path_csv_lookup = r"/home/mf1176/_repos/DetailView/lookup.csv"
path_csv_train  = r"/home/mf1176/_repos/DetailView/train_labels.csv"
#path_csv_vali   = r"/home/mf1176/_repos/DetailView/vali_labels.csv"
path_csv_test   = r"/mnt/gsdata//projects/ecosense/tls/single-pcs-tls-all/tree_heights.csv"
path_las        = r"/mnt/gsdata//projects/ecosense/tls/single-pcs-tls-all/"

# get mean & sd of height from training data
#train_metadata = pd.read_csv(path_csv_train)
train_height_mean = 15.20467594306311 #np.mean(train_metadata["tree_H"])
train_height_sd = 9.549458307689877 #np.std(train_metadata["tree_H"])

# add more threads
os.environ["OMP_NUM_THREADS"] = "40" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "40" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "40" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "40" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "40" # export NUMEXPR_NUM_THREADS=6
# Set the GPU to use (e.g., GPU 0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#%% make predictions

# load best model
# get the device
device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu")

print(device)

# model = net.ParallelDenseNet(n_classes = n_class, n_views = n_view)
model = net.SimpleView(n_classes = n_class, n_views = n_view)
model.load_state_dict(torch.load("/mnt/gsdata/projects/ecosense/DetailView-model/model_202305171452_60"))


# give to device
model.to(device)

# turn on evaluation mode
model.eval()

# setting up image augmentation
img_trans = transforms.Compose([
    transforms.RandomVerticalFlip(0.5)])

# prepare data for testing
test_dataset = net.TrainDataset_AllChannels(path_csv_test, path_las,  img_trans = img_trans, pc_rotate = True, height_noise = 0.01, test = True, res = res, n_sides = n_sides, height_mean = train_height_mean, height_sd = train_height_sd)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = int(n_batch/2), shuffle = False, pin_memory = True)

# create dictionary for the accumulated probabilities for each data point
all_paths = test_dataset.trees_frame.iloc[:,0]
data_probs = {path: [] for path in all_paths}

# Get the length of the dataset
dataset_length = len(test_dataset)
# Get the batch size from the DataLoader
batch_size = test_dataloader.batch_size

# Calculate the number of batches
num_batches = int(dataset_length / batch_size)

# iterate over the whole dataset 50 times (to get more stable predictions), for testing one or 2
for epoch in range(50):
    print("epoch: %d" % (epoch + 1))
    
    # iterate over validation dataloader in batches
    for i, t_data in enumerate(test_dataloader, 0):
        print(f"...epoch {epoch}:...{i}/{num_batches}")
        
        # load the batch
        t_inputs, t_heights, t_paths = t_data
        t_inputs, t_heights = t_inputs.to(device), t_heights.to(device)
        
        # get predictions
        t_preds = model(t_inputs, t_heights)
        t_probs = torch.nn.functional.softmax(t_preds, dim = 1)
        t_probs = t_probs.cpu().detach().numpy()
        
        # accumulate probabilities for each data point
        for i, path in enumerate(t_paths):
            if not any(data_probs[path]):
                data_probs[path] = t_probs[i,:]
            else:
                data_probs[path] += t_probs[i,:]

# get class id with maximum accumulated probabilities
max_prob_class = {}
for key, array in data_probs.items():
    max_prob_class[key] = np.argmax(array)

# create dataframe
df = pd.DataFrame({
    "filename": max_prob_class.keys(),
    "species_id": max_prob_class.values()})

# add species name
lookup = pd.read_csv(path_csv_lookup)
joined = pd.merge(df, lookup, on = 'species_id')
# joined = joined.drop("species_id", axis = 1)

# create a df with the probabilities
try:
    data_probs_df = pd.DataFrame.from_dict(data_probs, orient='index').reset_index()
    col_labels = lookup['species']
    data_probs_df.columns = pd.concat([pd.Series(["File"]), col_labels])
except Exception as e:
    print("There is an error in creating the probabilities dataframe")
    print(e)

# save data frame
output_dir = os.path.dirname(path_las)
joined.to_csv(os.path.join(output_dir, "predictions.csv"), index = False)
try:
    data_probs_df.to_csv(os.path.join(output_dir, "predictions_probabilities.csv"), index = False)
except Exception as e:
    print("No probabilities saved")
    print(e)