import os
import glob
import torch
import numpy as np
import pandas as pd
import pickle

def get_data(sensor_mode="coarse", gps_mask_ratio=0.0, imu_mask_ratio=0.0, split='train'):

    dir_path = f"./data/files/kitti_oxts/raw_data/{split}/"
    txt_files = glob.glob(os.path.join(dir_path, "*.txt"))
    with open("./data/files/kitti_oxts/raw_data/standard.pkl", "rb") as f:
        loaded_standardization = pickle.load(f)

    if sensor_mode == "coarse":
        results = {"gps": [], "imu": []}
    elif sensor_mode == "fine":
        results = {"gps_direct": [], "gps_indirect": [], "imu_direct": [], "imu_indirect": []}

    for file in txt_files:
        columns = [
            'lat', 'lon', 'alt',
            'roll', 'pitch', 'yaw',
            'vn', 've', 'vf', 'vl', 'vu',
            'ax', 'ay', 'az',
            'af', 'al', 'au',
            'wx', 'wy', 'wz',
            'wf', 'wl', 'wu',
            'pos_accuracy', 'vel_accuracy',
            'navstat', 'numsats',
            'posmode', 'velmode', 'orimode'
        ]
        data = np.loadtxt(file)
        df = pd.DataFrame(data=data, columns=columns)

        # GPS
        gps_col = ['lat', 'lon', 'alt', 'vn', 've', 'vf', 'vl', 'vu', 'pos_accuracy', 'vel_accuracy']
        gps_data = df[gps_col].copy()
        gps_data.loc[:, 'lat'] = ((gps_data['lat'] - gps_data['lat'].iloc[0]) - loaded_standardization["mean"]["lat"]) / loaded_standardization["std"]["lat"]
        gps_data.loc[:, 'lon'] = ((gps_data['lon'] - gps_data['lon'].iloc[0]) - loaded_standardization["mean"]["lon"]) / loaded_standardization["std"]["lon"]
        gps_data.loc[:, 'alt'] = ((gps_data['alt'] - gps_data['alt'].iloc[0]) - loaded_standardization["mean"]["alt"]) / loaded_standardization["std"]["alt"]

        gps_data.loc[:, 'vn'] = (gps_data['vn'] - loaded_standardization["mean"]["vn"]) / loaded_standardization["std"]["vn"]
        gps_data.loc[:, 've'] = (gps_data['ve'] - loaded_standardization["mean"]["ve"]) / loaded_standardization["std"]["ve"]
        gps_data.loc[:, 'vf'] = (gps_data['vf'] - loaded_standardization["mean"]["vf"]) / loaded_standardization["std"]["vf"]
        gps_mask = torch.Tensor((df['numsats'] >= 4)).reshape(-1, 1)

        # Apply random masking to GPS mask based on gps_mask_ratio
        if gps_mask_ratio > 0:
            num_elements = gps_mask.numel()
            num_zero_elements = int(num_elements * gps_mask_ratio)
            zero_indices = torch.randperm(num_elements)[:num_zero_elements]
            gps_mask.view(-1)[zero_indices] = 0

        # IMU
        imu_col = ['ax', 'ay', 'az', 'wx', 'wy', 'wz', 'roll', 'pitch', 'yaw', 'af', 'al', 'au', 'wf', 'wl', 'wu']
        imu_data = df[imu_col].copy()
        imu_data.loc[:, 'az'] = (imu_data['az'] - loaded_standardization["mean"]["az"]) / loaded_standardization["std"]["az"]
        imu_data.loc[:, 'wx'] = (imu_data['wx'] - loaded_standardization["mean"]["wx"]) / loaded_standardization["std"]["wx"]
        imu_data.loc[:, 'wy'] = (imu_data['wy'] - loaded_standardization["mean"]["wy"]) / loaded_standardization["std"]["wy"]
        imu_data.loc[:, 'pitch'] = (imu_data['pitch'] - loaded_standardization["mean"]["pitch"]) / loaded_standardization["std"]["pitch"]
        imu_data.loc[:, 'au'] = (imu_data['au'] - loaded_standardization["mean"]["au"]) / loaded_standardization["std"]["au"]
        imu_data.loc[:, 'wf'] = (imu_data['wf'] - loaded_standardization["mean"]["wf"]) / loaded_standardization["std"]["wf"]
        imu_data.loc[:, 'wl'] = (imu_data['wl'] - loaded_standardization["mean"]["wl"]) / loaded_standardization["std"]["wl"]
        imu_mask = torch.ones_like(gps_mask)

        # Apply random masking to IMU mask based on imu_mask_ratio
        if imu_mask_ratio > 0.0:
            num_elements = imu_mask.numel()
            num_zero_elements = int(num_elements * imu_mask_ratio)
            zero_indices = torch.randperm(num_elements)[:num_zero_elements]
            imu_mask.view(-1)[zero_indices] = 0

        if sensor_mode == "coarse":
            gps_data = torch.Tensor(gps_data.to_numpy())
            imu_data = torch.Tensor(imu_data.to_numpy())
            results['gps'].append({"data": gps_data * gps_mask, "mask": gps_mask})
            results['imu'].append({"data": imu_data * imu_mask, "mask": imu_mask})
        elif sensor_mode == "fine":
            gps_direct_col = ['lat', 'lon', 'alt', 'vn', 've']
            gps_indirect_col = ['vf', 'vl', 'vu', 'pos_accuracy', 'vel_accuracy']
            imu_direct_col = ['ax', 'ay', 'az', 'wx', 'wy', 'wz']
            imu_indirect_col = ['roll', 'pitch', 'yaw', 'af', 'al', 'au', 'wf', 'wl', 'wu']

            gps_direct = torch.Tensor(gps_data[gps_direct_col].to_numpy())
            gps_indirect = torch.Tensor(gps_data[gps_indirect_col].to_numpy())
            imu_direct = torch.Tensor(imu_data[imu_direct_col].to_numpy())
            imu_indirect = torch.Tensor(imu_data[imu_indirect_col].to_numpy())

            results['gps_direct'].append({"data": gps_direct * gps_mask, "mask": gps_mask})
            results['gps_indirect'].append({"data": gps_indirect * gps_mask, "mask": gps_mask})
            results['imu_direct'].append({"data": imu_direct * imu_mask, "mask": imu_mask})
            results['imu_indirect'].append({"data": imu_indirect * imu_mask, "mask": imu_mask})

    return results