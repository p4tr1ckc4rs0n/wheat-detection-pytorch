import torch
import pandas as pd

import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import utils
from dataset import WheatDataset, pre_process
from transforms import get_train_transform, get_valid_transform


def get_model_instance_segmentation(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


def main():
    # load and split data
    print('>>> splitting data into validation and training datasets')
    annotations = '/Users/Patrick/workspace/wheat/data/global-wheat-detection/train.csv'
    images_train_dir = '/Users/Patrick/workspace/wheat/data/global-wheat-detection/train'
    bbox_df = pd.read_csv(annotations)
    bbox_df_train, bbox_df_val = pre_process(bbox_df)

    # create pytorch train and validate datasets
    train_dataset = WheatDataset(bbox_df_train, images_train_dir, get_train_transform())
    valid_dataset = WheatDataset(bbox_df_val, images_train_dir, get_valid_transform())

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn
    )

    # load a model; pre-trained on COCO
    print('>>> loading model')
    device = torch.device('cpu')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = None
    num_epochs = 2
    itr = 1

    print('>>> begin training')
    for epoch in range(num_epochs):
        for images, targets, image_ids in train_data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if itr % 50 == 0:
                print(f"Iteration #{itr} loss: {loss_value}")
            itr += 1

        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

    torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')
    print('>>> saved model')


if __name__ == '__main__':
    main()
