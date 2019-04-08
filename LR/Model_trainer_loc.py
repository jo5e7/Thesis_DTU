import torch
from torch import nn, optim
import pandas as pd
from torchvision import transforms
from torchvision.transforms import Resize, RandomRotation, ToTensor, Normalize, RandomHorizontalFlip
from LR.Trainable_Model_LR import Trainable_Model_LR
from PADChest_DataLoading import PadChestDataset_loc
import Custome_losses
import argparse

import LR.DenseNet_LR as models

def get_densenet(target, target_loc, type=169):
    if type is 169:
        densenet = models.densenet_loc_169(pretrained=True)
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
    densenet.classifier = nn.Linear(num_ftrs, len(target))
    densenet.classifier_locations = nn.Linear(num_ftrs, len(target_loc))
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


    csv_train = r"C:\Users\maest\OneDrive\DTU\Semestre 4\Thesis\Code\CheXNet_aproach\Datase_stratification\PADChest_train_LRUMDP.csv"
    csv_test = r"C:\Users\maest\OneDrive\DTU\Semestre 4\Thesis\Code\CheXNet_aproach\Datase_stratification\PADChest_test_LRUMDP.csv"
    csv_val = r"C:\Users\maest\OneDrive\DTU\Semestre 4\Thesis\Code\CheXNet_aproach\Datase_stratification\PADChest_val_LRUMDP.csv"
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
    locations_labels = ['loc left', 'loc right', 'loc upper', 'loc middle', 'loc lower', 'loc pleural', 'loc mediastinum']

    # Transforms
    transforms_train = transforms.Compose([Resize(512), RandomHorizontalFlip(), RandomRotation(10), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transforms_test = transforms.Compose([Resize(512), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Create data loaders
    train_dataset = PadChestDataset_loc(csv_train, radiographic_findings_new, locations_labels, root_folder, transform=transforms_train, testing=True)
    test_dataset = PadChestDataset_loc(csv_test, radiographic_findings_new, locations_labels, root_folder, transform=transforms_test, testing=True)
    val_dataset = PadChestDataset_loc(csv_val, radiographic_findings_new, locations_labels, root_folder, transform=transforms_test, testing=True)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=True)
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
    densenet = get_densenet(radiographic_findings_new, locations_labels, type=169)
    Focal_non_zero = Custome_losses.FocalLoss_for_non_zero(logits=True, class_factor=0.25, groups=location_groups)
    sgd = optim.SGD(densenet.parameters(), lr=0.01, momentum=0.9)
    trainable_1 = Trainable_Model_LR(model=densenet, optimizer=sgd, loss_criterion_1=Focal,
                                     loss_criterion_2=Focal_non_zero, train_loader=train_dataloader,
                                     test_loader=test_dataloader, val_loader=val_dataloader, name='2_Focal_BCE_A0.25',
                                     description='alpha = 0.25')
    trainable_models.append(trainable_1)
    trainable_1.train_LR()

    densenet = get_densenet(radiographic_findings_new, locations_labels, type=169)
    Focal_non_zero = Custome_losses.FocalLoss_for_non_zero(logits=True, class_factor=0.25, groups=location_groups)
    adam = optim.Adam(densenet.parameters())
    trainable_1 = Trainable_Model_LR(model=densenet, optimizer=sgd, loss_criterion_1=Focal,
                                     loss_criterion_2=Focal_non_zero, train_loader=train_dataloader,
                                     test_loader=test_dataloader, val_loader=val_dataloader, name='2_Focal_Adam_A0.25',
                                     description='alpha = 0.25')
    trainable_models.append(trainable_1)
    trainable_1.train_LR()

    densenet = get_densenet(radiographic_findings_new, locations_labels, type=169)
    Focal_non_zero = Custome_losses.FocalLoss_for_non_zero(logits=True, class_factor=0.5, groups=location_groups)
    sgd = optim.SGD(densenet.parameters(), lr=0.01, momentum=0.9)
    trainable_1 = Trainable_Model_LR(model=densenet, optimizer=sgd, loss_criterion_1=Focal,
                                     loss_criterion_2=Focal_non_zero, train_loader=train_dataloader,
                                     test_loader=test_dataloader, val_loader=val_dataloader, name='2_Focal_BCE_A0.5',
                                     description='alpha = 0.5')
    trainable_models.append(trainable_1)
    trainable_1.train_LR()

    densenet = get_densenet(radiographic_findings_new, locations_labels, type=169)
    Focal_non_zero = Custome_losses.FocalLoss_for_non_zero(logits=True, class_factor=1, groups=location_groups)
    sgd = optim.SGD(densenet.parameters(), lr=0.01, momentum=0.9)
    trainable_1 = Trainable_Model_LR(model=densenet, optimizer=sgd, loss_criterion_1=Focal,
                                     loss_criterion_2=Focal_non_zero, train_loader=train_dataloader,
                                     test_loader=test_dataloader, val_loader=val_dataloader, name='2_Focal_BCE_A1',
                                     description='alpha = 1')
    trainable_models.append(trainable_1)
    trainable_1.train_LR()




    densenet = get_densenet(radiographic_findings_new, locations_labels, type=169)
    BCE_non_zero = Custome_losses.BCE_for_non_zero(logits=True, alpha=0.25, groups=location_groups)
    adam = optim.Adam(densenet.parameters())
    trainable_1 = Trainable_Model_LR(model=densenet, optimizer=adam, loss_criterion_1=BCE,
                                     loss_criterion_2=BCE_non_zero, train_loader=train_dataloader,
                                     test_loader=test_dataloader, val_loader=val_dataloader, name='2_BCE_Adam_A0.25',
                                     description='alpha = 0.25')
    trainable_models.append(trainable_1)
    trainable_1.train_LR()










