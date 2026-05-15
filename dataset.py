import os, torch, cv2, random
import numpy as np
from torch.utils.data import Dataset

class LRW1000Dataset(Dataset):
    def __init__(self, label_file, data_dir, transform=True):
        self.data = []
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = (96, 96)
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    self.data.append((parts[0], int(parts[1]), int(parts[2]), int(parts[3])))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        path, label, start, end = self.data[idx]
        cap = cv2.VideoCapture(os.path.join(self.data_dir, path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        flip = (random.random() > 0.5) if self.transform else False
        for _ in range(start, end + 1):
            ret, f = cap.read()
            if not ret: break
            f = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), self.target_size)
            if flip: f = cv2.flip(f, 1)
            frames.append(f.astype(np.float32) / 255.0)
        cap.release()
        if not frames: return None
        return torch.FloatTensor(np.array(frames)).unsqueeze(1), label
