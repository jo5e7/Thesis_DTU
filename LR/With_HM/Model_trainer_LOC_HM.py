import sys
sys.path.append('../../')
import torch
from torch import nn, optim
import pandas as pd
from torchvision import transforms
from torchvision.transforms import Resize, RandomRotation, ToTensor, Normalize, RandomHorizontalFlip
from LR.With_HM.Trainable_Model_LOC_HM import Trainable_Model_LR
from PADChest_DataLoading import PadChestDataset_loc, Resize_loc, RandomRotation_loc, ToTensor_loc, Normalize_loc, RandomHorizontalFlip_loc
import Custome_losses
import argparse

import LR.With_HM.DenseNet_LOC_HM as models


def get_WeightedRandomSampler(csv, single_class, testing=False):
    class_sample_counts = []
    df = pd.read_csv(csv)

    if testing:
        zero_df = df["ImageDir"] == 0
        df = df[zero_df]
        df = df.reset_index(drop=True)

    for c in range(len(single_class)):
        class_sample_counts.append(0)
        class_sample_counts.append(0)
        pass

    train_targets = []
    for i in range(len(df.index)):
        if df.loc[i, single_class[0]] == 1:
            class_sample_counts[1] += 1
            train_targets.append(1)
        else:
            class_sample_counts[0] += 1
            train_targets.append(0)
        pass

    print(len(df.index))
    print(single_class[0])
    print(class_sample_counts)

    weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    samples_weights = weights[train_targets]
    print(samples_weights, len(samples_weights))

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True)

    return sampler

    pass

def get_densenet(target, target_loc, type=169, all_in_1_reduction=False):
    if type is 169:
        densenet = models.densenet_loc_hm_169(pretrained=True)
        print("DenseNet_169")
    elif type is 161:
        densenet = models.densenet161(pretrained=True)
        print("DenseNet_161")
    elif type is 201:
        densenet = models.densenet201(pretrained=True)
        print("DenseNet_201")
    else:
        densenet = models.densenet_loc_121(pretrained=True)
        print("DenseNet_121")

    num_ftrs = densenet.classifier.in_features
    num_ftrs_loc = densenet.classifier_locations.in_features
    densenet.classifier = nn.Linear(num_ftrs, len(target))
    densenet.classifier_locations = nn.Linear(num_ftrs_loc, len(target_loc))
    densenet = densenet.cuda()
    densenet = torch.nn.DataParallel(densenet).cuda()
    return densenet


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root')
    parser.add_argument('--train_csv')
    parser.add_argument('--test_csv')
    parser.add_argument('--val_csv')
    parser.add_argument('--batch_size')
    args = parser.parse_args()

    root_folder = "D:\PADChest\images8"
    if vars(args)['root'] is not None:
        root_folder = vars(args)['root']

    csv_train = r"C:\Users\maest\OneDrive\DTU\Semestre 4\Thesis\Code\CheXNet_aproach\Datase_stratification\PADChest_train_LRUMDP_opacity.csv"
    csv_test = r"C:\Users\maest\OneDrive\DTU\Semestre 4\Thesis\Code\CheXNet_aproach\Datase_stratification\PADChest_test_LRUMDP_opacity.csv"
    csv_val = r"C:\Users\maest\OneDrive\DTU\Semestre 4\Thesis\Code\CheXNet_aproach\Datase_stratification\PADChest_val_LRUMDP_opacity.csv"

    if vars(args)['train_csv'] is not None:
        csv_train = vars(args)['train_csv']
        csv_test = vars(args)['test_csv']
        csv_val = vars(args)['val_csv']

    batch_size = 4
    if vars(args)['batch_size'] is not None:
        batch_size = vars(args)['batch_size']
        batch_size = int(batch_size)



    print(csv_train)
    print(len(pd.read_csv(csv_train).index))
    radiographic_findings_new = ['normal', 'calcified densities', 'nodule', 'fibrotic band', 'volume loss', 'pneumothorax', 'infiltrates', 'atelectasis', 'pleural thickening', 'pleural effusion', 'costophrenic angle blunting', 'cardiomegaly', 'aortic elongation', 'mediastinal enlargement', 'mass', 'thoracic cage deformation', 'fracture', 'hemidiaphragm elevation']
    radiographic_findings_opacity = ['opacity']
    locations_labels = ['loc left', 'loc right', 'loc upper', 'loc middle', 'loc lower', 'loc pleural', 'loc mediastinum']

    # Transforms
    transforms_train = transforms.Compose([Resize(512), RandomHorizontalFlip(), RandomRotation(10), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transforms_train_loc = transforms.Compose([Resize_loc(512), RandomHorizontalFlip_loc(), RandomRotation_loc(10), ToTensor_loc(), Normalize_loc(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transforms_test = transforms.Compose([Resize(512), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transforms_test_loc = transforms.Compose([Resize_loc(512), ToTensor_loc(), Normalize_loc(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Create data loaders
    train_dataset = PadChestDataset_loc(csv_train, radiographic_findings_opacity, locations_labels, root_folder, transform=transforms_train_loc, testing=False, pos_labels_always=True)
    test_dataset = PadChestDataset_loc(csv_test, radiographic_findings_opacity, locations_labels, root_folder, transform=transforms_test_loc, testing=False, pos_labels_always=True)
    val_dataset = PadChestDataset_loc(csv_val, radiographic_findings_opacity, locations_labels, root_folder, transform=transforms_test_loc, testing=False, pos_labels_always=True)

    #sampler = get_WeightedRandomSampler(csv_train, radiographic_findings_opacity, False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=True)
    #train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4,
    #                                               drop_last=True)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=True)


    # Loss functions

    BCE = nn.BCEWithLogitsLoss()

    Focal = Custome_losses.FocalLoss(logits=True)

    # Trainable
    trainable_models = []
    trainable_models_scores = []

    locations_labels = ['loc left', 'loc right', 'loc upper', 'loc middle', 'loc lower', 'loc pleural', 'loc mediastinum']
    location_groups = [1, 1, 2, 2, 2, 0, 0]


    #
    densenet = get_densenet(radiographic_findings_opacity, locations_labels, type=169, all_in_1_reduction=False)
    BCE_alpha = Custome_losses.BCE_alpha(logits=True, alpha=1.5)
    sgd = optim.SGD(densenet.parameters(), lr=0.001, momentum=0.9)
    trainable_1 = Trainable_Model_LR(model=densenet, optimizer=sgd, loss_criterion_1=BCE,
                                     loss_criterion_2=BCE_alpha, train_loader=train_dataloader,
                                     test_loader=test_dataloader, val_loader=val_dataloader,
                                     name='BCE_SGD0.001_a1.5_MLrAlwaysPos_BCEPos',
                                     description='alpha = 1.5')
    trainable_models.append(trainable_1)
    trainable_1.train_LR()

    densenet = get_densenet(radiographic_findings_opacity, locations_labels, type=169, all_in_1_reduction=False)
    BCE_alpha = Custome_losses.BCE_alpha(logits=True, alpha=0.5)
    sgd = optim.SGD(densenet.parameters(), lr=0.001, momentum=0.9)
    trainable_1 = Trainable_Model_LR(model=densenet, optimizer=sgd, loss_criterion_1=BCE,
                                     loss_criterion_2=BCE_alpha, train_loader=train_dataloader,
                                     test_loader=test_dataloader, val_loader=val_dataloader,
                                     name='BCE_SGD0.001_a0.5_MLrAlwaysPos_BCEPos',
                                     description='alpha = 0.5')
    trainable_models.append(trainable_1)
    trainable_1.train_LR()

    densenet = get_densenet(radiographic_findings_opacity, locations_labels, type=169, all_in_1_reduction=False)
    BCE_alpha = Custome_losses.BCE_alpha(logits=True, alpha=0.25)
    sgd = optim.SGD(densenet.parameters(), lr=0.001, momentum=0.9)
    trainable_1 = Trainable_Model_LR(model=densenet, optimizer=sgd, loss_criterion_1=BCE,
                                     loss_criterion_2=BCE_alpha, train_loader=train_dataloader,
                                     test_loader=test_dataloader, val_loader=val_dataloader,
                                     name='BCE_SGD0.001_a0.25_MLrAlwaysPos_BCEPos',
                                     description='alpha = 0.25')
    trainable_models.append(trainable_1)
    trainable_1.train_LR()

    densenet = get_densenet(radiographic_findings_opacity, locations_labels, type=169, all_in_1_reduction=False)
    BCE_alpha = Custome_losses.BCE_alpha(logits=True, alpha=0.1)
    sgd = optim.SGD(densenet.parameters(), lr=0.001, momentum=0.9)
    trainable_1 = Trainable_Model_LR(model=densenet, optimizer=sgd, loss_criterion_1=BCE,
                                     loss_criterion_2=BCE_alpha, train_loader=train_dataloader,
                                     test_loader=test_dataloader, val_loader=val_dataloader,
                                     name='BCE_SGD0.001_a0.1_MLrAlwaysPos_BCEPos',
                                     description='alpha = 0.1')
    trainable_models.append(trainable_1)
    trainable_1.train_LR()





