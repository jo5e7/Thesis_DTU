import torch
from torch import nn, optim
import pandas as pd
import numpy as np
import torchvision
from torchvision import transforms
from torchvision.transforms import Resize, RandomRotation, ToTensor, Normalize, RandomHorizontalFlip
from Baseline.With_HM.Trainable_model import TrainableModel
from PADChest_DataLoading import PadChestDataset
import Custome_losses
import argparse
from Baseline.With_HM import DenseNet_HM


def get_training_weights(labels, df, data_lenght):
    labels_count = []
    for i in range(len(labels)):
        labels_count.append((df.iloc[0][labels[i]]))

    positive_samples_weights = []
    classes_weights = []
    for i in range(len(labels)):
        count = labels_count[i]
        positive_samples_weights.append((data_lenght - count) / count)
    print(positive_samples_weights)

    return positive_samples_weights

def create_labels_info(df, labels):
    dict_labels_info = {}
    for l in labels:
        dict_labels_info[l] = 0

    df_length = len(df.index)
    for i in range(df_length):
        print(i/df_length)
        for l in labels:
            if df.loc[i, l] == 1:
                dict_labels_info[l] += 1

    print(dict_labels_info)
    return pd.DataFrame(dict_labels_info, index=[0])

def get_densenet(target, type=121):
    if type is 169:
        densenet = DenseNet_HM.DenseNet()
        print("DenseNet_169")
    elif type is 161:
        densenet = torchvision.models.densenet161(pretrained=True)
        print("DenseNet_161")
    elif type is 201:
        densenet = torchvision.models.densenet201(pretrained=True)
        print("DenseNet_201")
    else:
        densenet = torchvision.models.densenet121(pretrained=True)
        print("DenseNet_121")

    num_ftrs = densenet.classifier.in_features
    densenet.classifier = nn.Linear(166400, len(target))
    densenet = densenet.cuda()
    densenet = torch.nn.DataParallel(densenet).cuda()
    return densenet


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root')
    parser.add_argument('--train_csv')
    parser.add_argument('--test_csv')
    parser.add_argument('--val_csv')
    parser.add_argument('--hm_csv')
    parser.add_argument('--batch_size')
    args = parser.parse_args()

    root_folder = "D:\PADChest\images8"
    if vars(args)['root'] is not None:
        root_folder = vars(args)['root']


    csv_train = r"C:\Users\maest\OneDrive\DTU\Semestre 4\Thesis\Code\CheXNet_aproach\Datase_stratification\PADChest_train_LRUMDP_opacity.csv"
    csv_test = r"C:\Users\maest\OneDrive\DTU\Semestre 4\Thesis\Code\CheXNet_aproach\Datase_stratification\PADChest_test_LRUMDP_opacity.csv"
    csv_val = r"C:\Users\maest\OneDrive\DTU\Semestre 4\Thesis\Code\CheXNet_aproach\Datase_stratification\PADChest_val_LRUMDP_opacity.csv"
    csv_hm = r"C:\Users\maest\OneDrive\DTU\Semestre 4\Thesis\Code\CheXNet_aproach\Datase_stratification\PADChest_hm_LRUMDP_opacity.csv"
    if vars(args)['train_csv'] is not None:
        csv_train = vars(args)['train_csv']
        csv_test = vars(args)['test_csv']
        csv_val = vars(args)['val_csv']
        csv_hm = vars(args)['hm_csv']

    batch_size = 4
    if vars(args)['batch_size'] is not None:
        batch_size = vars(args)['batch_size']
        batch_size = int(batch_size)



    print(csv_train)
    print(len(pd.read_csv(csv_train).index))
    radiographic_findings_all = ['normal', 'calcified densities', 'nodule', 'fibrotic band', 'volume loss', 'pneumothorax', 'air trapping', 'bronchiectasis', 'infiltrates', 'atelectasis', 'pleural thickening', 'pleural effusion', 'costophrenic angle blunting', 'hilar enlargement', 'cardiomegaly', 'aortic elongation', 'mediastinal enlargement', 'mass', 'thoracic cage deformation', 'vertebral degenerative changes', 'fracture', 'hemidiaphragm elevation']
    radiographic_findings_new = ['normal', 'calcified densities', 'nodule', 'fibrotic band', 'volume loss', 'pneumothorax', 'infiltrates', 'atelectasis', 'pleural thickening', 'pleural effusion', 'costophrenic angle blunting', 'cardiomegaly', 'aortic elongation', 'mediastinal enlargement', 'mass', 'thoracic cage deformation', 'fracture', 'hemidiaphragm elevation']
    radiographic_findings_opacity = ['opacity']


    # Create positive samples weights
    positive_samples_w = get_training_weights(radiographic_findings_opacity, create_labels_info(pd.read_csv(csv_train), radiographic_findings_opacity), len(pd.read_csv(csv_train).index))
    print(np.array(positive_samples_w))

    # Transforms
    transforms_train = transforms.Compose([Resize(512), RandomHorizontalFlip(), RandomRotation(10), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transforms_test = transforms.Compose([Resize(512), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Create data loaders
    train_dataset = PadChestDataset(csv_train, radiographic_findings_opacity, root_folder, transform=transforms_train, testing=False)
    test_dataset = PadChestDataset(csv_test, radiographic_findings_opacity, root_folder, transform=transforms_test, testing=False)
    val_dataset = PadChestDataset(csv_val, radiographic_findings_opacity, root_folder, transform=transforms_test, testing=False)
    hm_dataset = PadChestDataset(csv_hm, radiographic_findings_opacity, root_folder, transform=transforms_test)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=True)
    hm_dataloader = torch.utils.data.DataLoader(hm_dataset, batch_size=1)

    # Create Backbones




    # Loss functions
    BCE_pos_weights = torch.FloatTensor(positive_samples_w)
    if torch.cuda.is_available:
        BCE_pos_weights = BCE_pos_weights.cuda()
    BCE = nn.BCEWithLogitsLoss()
    BCE_w = nn.BCEWithLogitsLoss(pos_weight=BCE_pos_weights)

    Focal = Custome_losses.FocalLoss(logits=True)
    Focal_w = Custome_losses.FocalLoss(pos_weight=BCE_pos_weights, logits=True)

    # Trainable
    trainable_models = []
    trainable_models_scores = []

    #
    densenet = get_densenet(radiographic_findings_opacity, type=169)
    sgd = optim.SGD(densenet.parameters(), lr=0.01, momentum=0.9)
    trainable_1 = TrainableModel(model=densenet, optimizer=sgd, loss_criterion=BCE,
                                 train_loader=train_dataloader,
                                 test_loader=test_dataloader, val_loader=val_dataloader, hm_loader=hm_dataloader, name='Opacity_DN169_BCE_SGD')
    trainable_models.append(trainable_1)
    trainable_1.train()

    densenet = get_densenet(radiographic_findings_opacity, type=169)
    sgd = optim.SGD(densenet.parameters(), lr=0.01, momentum=0.9)
    trainable_1 = TrainableModel(model=densenet, optimizer=sgd, loss_criterion=Focal,
                                 train_loader=train_dataloader,
                                 test_loader=test_dataloader, val_loader=val_dataloader, hm_loader=hm_dataloader, name='Opacity_DN169_FL_SGD')
    trainable_models.append(trainable_1)
    #trainable_1.train()







