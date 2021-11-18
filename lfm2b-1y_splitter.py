import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

INF_STR = "{:10d} entries {:7d} users {:7d} items for {}"

parser = argparse.ArgumentParser()

parser.add_argument('--listening_history_path', '-lh', type=str,
                    help="Path to 'users.tsv' and 'listening_events.tsv' of the LFM2b-1y dataset.")
parser.add_argument('--saving_path', '-s', type=str, help="Path where to save the split data. Default to './'",
                    default='./')

args = parser.parse_args()

listening_history_path = args.listening_history_path
saving_path = args.saving_path
k = 1

user_info_path = os.path.join(listening_history_path, 'users.tsv')
item_info_path = os.path.join(listening_history_path, 'tracks.tsv')
listening_events_path = os.path.join(listening_history_path, 'listening_events.tsv')

lhs = pd.read_csv(listening_events_path, sep='\t', names=['old_user_id', 'old_item_id', 'timestamp'], skiprows=1)
print(INF_STR.format(len(lhs), lhs.old_user_id.nunique(), lhs.old_item_id.nunique(), 'Original Data'))

# We keep only the data from the last month
lhs = lhs[lhs.timestamp >= '2020-02-20 00:00:00']
print(INF_STR.format(len(lhs), lhs.old_user_id.nunique(), lhs.old_item_id.nunique(), 'Only last month'))

# Loading users and items
users = pd.read_csv(user_info_path, delimiter='\t', names=['old_user_id', 'gender', 'country', 'age', 'creation_time'],
                    skiprows=1)

# Only users with gender in m/f and 10 <= age <= 95
users = users[(users.gender.isin(['m', 'f'])) & (users.age >= 10) & (users.age <= 95)]
lhs = lhs[lhs.old_user_id.isin(set(users.old_user_id))]
print(INF_STR.format(len(lhs), lhs.old_user_id.nunique(), lhs.old_item_id.nunique(),
                     'Only users with gender and valid age'))

# Keeping only the first interaction
lhs = lhs.sort_values('timestamp')
lhs = lhs.drop_duplicates(subset=['old_user_id', 'old_item_id'])
print(INF_STR.format(len(lhs), lhs.old_user_id.nunique(), lhs.old_item_id.nunique(),
                     'Keeping only the first interaction'))

# Removing power users
user_counts = lhs.old_user_id.value_counts()
perc_99 = np.percentile(user_counts, 99)
user_below = set(user_counts[user_counts <= perc_99].index)
lhs = lhs[lhs.old_user_id.isin(user_below)]

print(INF_STR.format(len(lhs), lhs.old_user_id.nunique(), lhs.old_item_id.nunique(),
                     'Removed power users (below the 99% percentile)'))

# 5-core filtering
while True:
    start_number = len(lhs)

    # Item pass
    item_counts = lhs.old_item_id.value_counts()
    item_above = set(item_counts[item_counts >= 5].index)
    lhs = lhs[lhs.old_item_id.isin(item_above)]
    print('Records after item pass: ', len(lhs))

    # User pass
    user_counts = lhs.old_user_id.value_counts()
    user_above = set(user_counts[user_counts >= 5].index)
    lhs = lhs[lhs.old_user_id.isin(user_above)]
    print('Records after user pass: ', len(lhs))

    if len(lhs) == start_number:
        print('Exiting...')
        break

print(INF_STR.format(len(lhs), lhs.old_user_id.nunique(), lhs.old_item_id.nunique(), '10-core filtering'))

# Creating simple integer indexes used for sparse matrices
user_ids = lhs.old_user_id.drop_duplicates().reset_index(drop=True)
item_ids = lhs.old_item_id.drop_duplicates().reset_index(drop=True)
user_ids.index.name = 'user_id'
item_ids.index.name = 'item_id'
user_ids = user_ids.reset_index()
item_ids = item_ids.reset_index()
lhs = lhs.merge(user_ids).merge(item_ids)

print('Splitting the data, temporal ordered - leave-k-out.')

lhs = lhs.sort_values('timestamp')
train_idxs = []
val_idxs = []
test_idxs = []
for user, user_group in tqdm(lhs.groupby('old_user_id')):
    # Data is already sorted by timestamp
    if len(user_group) <= k * 2:
        # Not enough data for val/test data. Place the user in train.
        train_idxs += (list(user_group.index))
    else:
        train_idxs += list(user_group.index[:-2 * k])
        val_idxs += list(user_group.index[-2 * k:-k])
        test_idxs += list(user_group.index[-k:])

train_data = lhs.loc[train_idxs]
val_data = lhs.loc[val_idxs]
test_data = lhs.loc[test_idxs]

print(INF_STR.format(len(train_data), train_data.old_user_id.nunique(), train_data.old_item_id.nunique(), 'Train Data'))
print(INF_STR.format(len(val_data), val_data.old_user_id.nunique(), val_data.old_item_id.nunique(), 'Val Data'))
print(INF_STR.format(len(test_data), test_data.old_user_id.nunique(), test_data.old_item_id.nunique(), 'Test Data'))

# Saving locally

print('Saving data to {}'.format(saving_path))

train_data.to_csv(os.path.join(saving_path, 'listening_history_train.csv'), index=False)
val_data.to_csv(os.path.join(saving_path, 'listening_history_val.csv'), index=False)
test_data.to_csv(os.path.join(saving_path, 'listening_history_test.csv'), index=False)

user_ids.to_csv(os.path.join(saving_path, 'user_ids.csv'), index=False)
item_ids.to_csv(os.path.join(saving_path, 'item_ids.csv'), index=False)
