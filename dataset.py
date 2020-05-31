import cv2
import torch
import numpy as np
from scipy import stats
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class WheatDataset(Dataset):
    def __init__(self, bbox_df, image_dir, transforms=None):
        self._img_dir = image_dir
        self._bbox_df = bbox_df
        self._image_ids = self._bbox_df['image_id'].unique()
        self._transforms = transforms

    def __getitem__(self, idx):
        # load image
        image_id = self._image_ids[idx]
        image = cv2.imread(f'{self._img_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        # load bounding boxes
        image_bbox = self._bbox_df[self._bbox_df['image_id'] == image_id]
        num_boxes = len(image_bbox)
        # get bounding box coordinates
        img_bboxes = image_bbox[['x', 'y', 'x1', 'y1']].values
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(img_bboxes, dtype=torch.float32)
        areas = image_bbox['area'].values
        # there is only one class
        labels = torch.ones((num_boxes,), dtype=torch.int64)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_boxes,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = areas
        target["image_id"] = torch.tensor([idx])
        target["iscrowd"] = iscrowd

        if self._transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self._transforms(**sample)
            image = sample['image']

            target['boxes'] = torch.tensor(sample['bboxes'])
            return image, target, image_id

    def __len__(self):
        return len(self._image_ids)


def pre_process(df, threshold=3):
    df[['x', 'y', 'w', 'h']] = (df['bbox']
                                .str
                                .strip('[]')
                                .str
                                .split(',', expand=True)
                                .astype(np.float32))
    df['x1'] = df['x'] + df['w']
    df['y1'] = df['y'] + df['h']
    df['area'] = df['w'] * df['h']
    df['z-score'] = np.abs(stats.zscore(df['area']))
    df = df[df['z-score'] < threshold]
    train, validate = train_test_split(df, test_size=0.2, random_state=42)
    return train, validate
