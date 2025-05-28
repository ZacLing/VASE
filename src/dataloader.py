import torch
from torch.utils.data import Dataset, DataLoader

def pad_tensor(tensor, target_length, pad_value=0, device='cpu'):
    """
    将输入tensor在第0维上填充至target_length长度。
    如果tensor长度不足，则在后面补上pad_value。

    参数：
        tensor: 待填充的tensor，形状为(n, ...)；
        target_length: 目标长度；
        pad_value: 填充值，默认0。
        device: 目标设备，默认'cpu'。

    返回：
        填充后的tensor，形状为(target_length, ...)。
    """
    cur_length = tensor.size(0)
    if cur_length >= target_length:
        # 如果当前长度大于等于目标长度，则截断
        return tensor[:target_length].to(device)
    else:
        # 计算需要填充的长度
        pad_shape = list(tensor.shape)
        pad_shape[0] = target_length - cur_length
        # 生成填充值的tensor
        pad_tensor_ = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=device)
        # 在后面拼接
        return torch.cat([tensor, pad_tensor_], dim=0)

class SensorDataset(Dataset):
    def __init__(self, x, label, max_length, stride, drift, device='cpu'):
        """
        构建自定义数据集类，用于支持DataLoader。
        参数：
            x: dict，每个key为sensor名称，对应的value为list，
               list中每个元素为一个子dict，包含：
                    "data": Tensor(seq_len, input_dim)
                    "mask": Tensor(seq_len, 1)
            label: dict，每个key为sensor名称，对应的value为list，
               list中每个元素为一个子dict，包含：
                    "data": Tensor(seq_len, input_dim)
                    "mask": Tensor(seq_len, 1)
            max_length: int，将每个子dict中的序列切分为窗口的最大长度。
                        如果不足max_length，则用0填充，同时填充部分mask置为0。
            stride: int，滑动窗口的步长。
            drift: int，labels相对于inputs的偏移量，即labels的第i个元素来自inputs的第i+drift个位置，
                   如果超出原序列，则填充0。
            device: str，指定数据和模型的设备，默认'cpu'。
        """
        self.x = x
        self.label = label
        self.max_length = max_length
        self.stride = stride
        self.drift = drift
        self.device = device
        self.key = []

        # 保存每个sensor所有切分后的窗口，窗口内包含输入和标签（数据及mask）
        self.windows_by_sensor = self._process_data()

    def _process_data(self):
        """
        处理数据，使用滑动窗口切分序列并生成窗口。
        """
        windows_by_sensor = {}
        # 遍历每个sensor
        for sensor_key, sample_list in self.x.items():
            self.key.append(sensor_key)
            sensor_windows = []
            # 遍历该sensor下的每个子dict（独立的序列）
            for x_sample, label_sample in zip(sample_list, self.label[sensor_key]):
                x_data = x_sample["data"].to(self.device)   # Tensor，形状 (seq_len, input_dim)
                x_mask = x_sample["mask"].to(self.device)   # Tensor，形状 (seq_len, 1)
                label_data = label_sample["data"].to(self.device)   # Tensor，形状 (seq_len, input_dim)
                label_mask = label_sample["mask"].to(self.device)   # Tensor，形状 (seq_len, 1)
                seq_len = x_data.size(0)

                # 对当前子dict使用滑动窗口切分，注意各子dict之间相互独立
                start_idx = 0
                while start_idx < seq_len:
                    end_idx = start_idx + self.max_length
                    # 取出inputs部分，并进行pad操作
                    input_data = x_data[start_idx: end_idx]
                    input_mask = x_mask[start_idx: end_idx]
                    input_data = pad_tensor(input_data, self.max_length, pad_value=0, device=self.device)
                    input_mask = pad_tensor(input_mask, self.max_length, pad_value=0, device=self.device)

                    # 构造labels窗口：相当于inputs整体向后平移drift个位置
                    label_start = start_idx + self.drift
                    label_end = label_start + self.max_length
                    # 若label_start超过序列长度，则返回空tensor，后续将被pad
                    if label_start < seq_len:
                        label_data_window = label_data[label_start: label_end]
                        label_mask_window = label_mask[label_start: label_end]
                    else:
                        label_data_window = label_data[0:0].to(self.device)  # 空tensor
                        label_mask_window = label_mask[0:0].to(self.device)
                    label_data_window = pad_tensor(label_data_window, self.max_length, pad_value=0, device=self.device)
                    label_mask_window = pad_tensor(label_mask_window, self.max_length, pad_value=0, device=self.device)

                    # 保存当前窗口，包含inputs和labels
                    sensor_windows.append({
                        "input_data": input_data,
                        "input_mask": input_mask,
                        "label_data": label_data_window,
                        "label_mask": label_mask_window
                    })

                    start_idx += self.stride
            windows_by_sensor[sensor_key] = sensor_windows

        return windows_by_sensor

    def __len__(self):
        """
        返回数据集中的总样本数（窗口数目）。
        """
        # 获取所有sensor的窗口数目，确保每个sensor的窗口数目相同
        sensor_keys = list(self.x.keys())
        num_windows_list = [len(self.windows_by_sensor[sensor]) for sensor in sensor_keys]
        return min(num_windows_list)

    def __getitem__(self, idx):
        """
        获取指定索引的样本。
        """
        batch_inputs = {}
        batch_labels = {}
        # 遍历每个sensor，提取对应idx的窗口
        for sensor_key in self.windows_by_sensor:
            sensor_windows = self.windows_by_sensor[sensor_key]
            sample = sensor_windows[idx]

            batch_inputs[sensor_key] = {"data": sample["input_data"], "mask": sample["input_mask"]}
            batch_labels[sensor_key] = {"data": sample["label_data"], "mask": sample["label_mask"]}

        return batch_inputs, batch_labels

    def keys(self):
        return self.key


def create_dataloader(x, label, max_length, stride, drift, batch_size, device='cpu'):
    """
    使用自定义数据集类创建DataLoader。
    """
    sensor_dataset = SensorDataset(x, label, max_length, stride, drift, device)
    return DataLoader(sensor_dataset, batch_size=batch_size, shuffle=True)