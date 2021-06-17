import os
import shutil

train_path = 'trainval/'
gt_out = 'gt_val/'

for dirs in os.listdir(train_path):
    newdir = os.path.join(train_path, dirs)
    print('start processing ' + dirs)
    new_name = gt_out + dirs + '.txt'
    old_name = newdir + '/groundtruth.txt'
    shutil.copy(old_name, new_name)
