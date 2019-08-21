import torch
from torch import nn, optim
import pandas as pd
from torchvision import transforms
from torchvision.transforms import Resize, RandomRotation, ToTensor, Normalize, RandomHorizontalFlip
from Attention.Trainable_Model_att import Trainable_Model_Att
from PADChest_DataLoading import PadChestDataset_loc, Resize_loc, RandomRotation_loc, ToTensor_loc, Normalize_loc, RandomHorizontalFlip_loc
import Custome_losses
import argparse

import Attention.DenseNet_Att as models

def get_densenet_att(target, target_loc, type=169, bp_elementwise=False):
    if type is 169:
        densenet_att = models.densenet_att_169(pretrained=True, bp_elementwise=bp_elementwise)
        print("DenseNet_169")
    elif type is 161:
        densenet_att = models.densenet161(pretrained=True)
        print("DenseNet_161")
    elif type is 201:
        densenet_att = models.densenet201(pretrained=True)
        print("DenseNet_201")
    else:
        densenet_att = models.densenet_att_121(pretrained=True, bp_elementwise=bp_elementwise)
        print("DenseNet_121")

    num_ftrs = densenet_att.classifier.in_features
    densenet_att.classifier = nn.Linear(num_ftrs, len(target))
    densenet_att.classifier_locations = nn.Linear(num_ftrs, len(target_loc))
    densenet_att = densenet_att.cuda()
    densenet_att = torch.nn.DataParallel(densenet_att).cuda()
    return densenet_att

def get_densenet_multi_att(target, target_loc, type=169, bp_position=False, all_in_1_reduction=False, hidden_layers_att=1):
    if type is 169:
        densenet_MA = models.densenet_multi_att_169(pretrained=True, bp_position=bp_position, all_in_1_reduction=all_in_1_reduction, hidden_layers_att=hidden_layers_att)
        print("DenseNet_169")
    elif type is 161:
        densenet_MA = models.densenet161(pretrained=True)
        print("DenseNet_161")
    elif type is 201:
        densenet_MA = models.densenet201(pretrained=True)
        print("DenseNet_201")
    else:
        densenet_MA = models.densenet_multi_att_121(pretrained=True, bp_elementwise=True)
        print("DenseNet_121")

    num_ftrs = densenet_MA.classifier.in_features
    num_ftrs_loc = densenet_MA.classifier_locations.in_features
    densenet_MA.classifier = nn.Linear(num_ftrs, len(target))
    densenet_MA.classifier_locations = nn.Linear(num_ftrs_loc, len(target_loc))
    densenet_MA = densenet_MA.cuda()
    densenet_MA = torch.nn.DataParallel(densenet_MA).cuda()
    return densenet_MA


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
    transforms_train = transforms.Compose([Resize(512), RandomHorizontalFlip(), RandomRotation(10), ToTensor(),
                                           Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transforms_train_loc = transforms.Compose(
        [Resize_loc(512), RandomHorizontalFlip_loc(), RandomRotation_loc(10), ToTensor_loc(),
         Normalize_loc(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transforms_test = transforms.Compose(
        [Resize(512), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transforms_test_loc = transforms.Compose(
        [Resize_loc(512), ToTensor_loc(), Normalize_loc(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Create data loaders
    train_dataset = PadChestDataset_loc(csv_train, radiographic_findings_opacity, locations_labels, root_folder, transform=transforms_train_loc, testing=False)
    test_dataset = PadChestDataset_loc(csv_test, radiographic_findings_opacity, locations_labels, root_folder, transform=transforms_test_loc, testing=False)
    val_dataset = PadChestDataset_loc(csv_val, radiographic_findings_opacity, locations_labels, root_folder, transform=transforms_test_loc, testing=False)

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
    densenet = get_densenet_multi_att(radiographic_findings_opacity, locations_labels, type=169, bp_position=True,
                                      all_in_1_reduction=False)
    BCE_non_zero = Custome_losses.BCE_for_non_zero(logits=True, alpha=1, groups=location_groups)
    sgd = optim.SGD(densenet.parameters(), lr=0.003, momentum=0.9)
    trainable_1 = Trainable_Model_Att(model=densenet, optimizer=sgd, loss_criterion_1=BCE,
                                      loss_criterion_2=BCE_non_zero, train_loader=train_dataloader,
                                      test_loader=test_dataloader, val_loader=val_dataloader,
                                      name='BCE_SGD0.003_a1_opacity_MAr_BPpos',
                                      description='alpha = 1')
    trainable_models.append(trainable_1)
    trainable_1.train_Att()

    densenet = get_densenet_multi_att(radiographic_findings_opacity, locations_labels, type=169, bp_position=True,
                                      all_in_1_reduction=False)
    BCE_non_zero = Custome_losses.BCE_for_non_zero(logits=True, alpha=0.5, groups=location_groups)
    sgd = optim.SGD(densenet.parameters(), lr=0.003, momentum=0.9)
    trainable_1 = Trainable_Model_Att(model=densenet, optimizer=sgd, loss_criterion_1=BCE,
                                      loss_criterion_2=BCE_non_zero, train_loader=train_dataloader,
                                      test_loader=test_dataloader, val_loader=val_dataloader,
                                      name='BCE_SGD0.003_a0.5_opacity_MAr_BPpos',
                                      description='alpha = 0.5')
    trainable_models.append(trainable_1)
    trainable_1.train_Att()

    densenet = get_densenet_multi_att(radiographic_findings_opacity, locations_labels, type=169, bp_position=True,
                                      all_in_1_reduction=False)
    BCE_non_zero = Custome_losses.BCE_for_non_zero(logits=True, alpha=0.1, groups=location_groups)
    sgd = optim.SGD(densenet.parameters(), lr=0.003, momentum=0.9)
    trainable_1 = Trainable_Model_Att(model=densenet, optimizer=sgd, loss_criterion_1=BCE,
                                      loss_criterion_2=BCE_non_zero, train_loader=train_dataloader,
                                      test_loader=test_dataloader, val_loader=val_dataloader,
                                      name='BCE_SGD0.003_a0.1_opacity_MAr_BPpos',
                                      description='alpha = 0.1')
    trainable_models.append(trainable_1)
    trainable_1.train_Att()



#
    densenet = get_densenet_multi_att(radiographic_findings_opacity, locations_labels, type=169, bp_position=True,
                                      all_in_1_reduction=False, hidden_layers_att=2)
    BCE_non_zero = Custome_losses.BCE_for_non_zero(logits=True, alpha=1, groups=location_groups)
    sgd = optim.SGD(densenet.parameters(), lr=0.003, momentum=0.9)
    trainable_1 = Trainable_Model_Att(model=densenet, optimizer=sgd, loss_criterion_1=BCE,
                                      loss_criterion_2=BCE_non_zero, train_loader=train_dataloader,
                                      test_loader=test_dataloader, val_loader=val_dataloader,
                                      name='BCE_SGD0.003_a1_opacity_MAr_BPpos_HLA2',
                                      description='alpha = 1')
    trainable_models.append(trainable_1)
    trainable_1.train_Att()

    densenet = get_densenet_multi_att(radiographic_findings_opacity, locations_labels, type=169, bp_position=True,
                                      all_in_1_reduction=False, hidden_layers_att=2)
    BCE_non_zero = Custome_losses.BCE_for_non_zero(logits=True, alpha=0.5, groups=location_groups)
    sgd = optim.SGD(densenet.parameters(), lr=0.003, momentum=0.9)
    trainable_1 = Trainable_Model_Att(model=densenet, optimizer=sgd, loss_criterion_1=BCE,
                                      loss_criterion_2=BCE_non_zero, train_loader=train_dataloader,
                                      test_loader=test_dataloader, val_loader=val_dataloader,
                                      name='BCE_SGD0.003_a0.5_opacity_MAr_BPpos_HLA2',
                                      description='alpha = 0.5')
    trainable_models.append(trainable_1)
    trainable_1.train_Att()

    densenet = get_densenet_multi_att(radiographic_findings_opacity, locations_labels, type=169, bp_position=True,
                                      all_in_1_reduction=False, hidden_layers_att=2)
    BCE_non_zero = Custome_losses.BCE_for_non_zero(logits=True, alpha=0.1, groups=location_groups)
    sgd = optim.SGD(densenet.parameters(), lr=0.003, momentum=0.9)
    trainable_1 = Trainable_Model_Att(model=densenet, optimizer=sgd, loss_criterion_1=BCE,
                                      loss_criterion_2=BCE_non_zero, train_loader=train_dataloader,
                                      test_loader=test_dataloader, val_loader=val_dataloader,
                                      name='BCE_SGD0.003_a0.1_opacity_MAr_BPpos_HLA2',
                                      description='alpha = 0.1')
    trainable_models.append(trainable_1)
    trainable_1.train_Att()


#
    densenet = get_densenet_multi_att(radiographic_findings_opacity, locations_labels, type=169, bp_position=True,
                                      all_in_1_reduction=False, hidden_layers_att=4)
    BCE_non_zero = Custome_losses.BCE_for_non_zero(logits=True, alpha=1, groups=location_groups)
    sgd = optim.SGD(densenet.parameters(), lr=0.003, momentum=0.9)
    trainable_1 = Trainable_Model_Att(model=densenet, optimizer=sgd, loss_criterion_1=BCE,
                                      loss_criterion_2=BCE_non_zero, train_loader=train_dataloader,
                                      test_loader=test_dataloader, val_loader=val_dataloader,
                                      name='BCE_SGD0.003_a1_opacity_MAr_BPpos_HLA4',
                                      description='alpha = 1')
    trainable_models.append(trainable_1)
    trainable_1.train_Att()

    densenet = get_densenet_multi_att(radiographic_findings_opacity, locations_labels, type=169, bp_position=True,
                                      all_in_1_reduction=False, hidden_layers_att=4)
    BCE_non_zero = Custome_losses.BCE_for_non_zero(logits=True, alpha=0.5, groups=location_groups)
    sgd = optim.SGD(densenet.parameters(), lr=0.003, momentum=0.9)
    trainable_1 = Trainable_Model_Att(model=densenet, optimizer=sgd, loss_criterion_1=BCE,
                                      loss_criterion_2=BCE_non_zero, train_loader=train_dataloader,
                                      test_loader=test_dataloader, val_loader=val_dataloader,
                                      name='BCE_SGD0.003_a0.5_opacity_MAr_BPpos_HLA4',
                                      description='alpha = 0.5')
    trainable_models.append(trainable_1)
    trainable_1.train_Att()

    densenet = get_densenet_multi_att(radiographic_findings_opacity, locations_labels, type=169, bp_position=True,
                                      all_in_1_reduction=False, hidden_layers_att=4)
    BCE_non_zero = Custome_losses.BCE_for_non_zero(logits=True, alpha=0.1, groups=location_groups)
    sgd = optim.SGD(densenet.parameters(), lr=0.003, momentum=0.9)
    trainable_1 = Trainable_Model_Att(model=densenet, optimizer=sgd, loss_criterion_1=BCE,
                                      loss_criterion_2=BCE_non_zero, train_loader=train_dataloader,
                                      test_loader=test_dataloader, val_loader=val_dataloader,
                                      name='BCE_SGD0.003_a0.1_opacity_MAr_BPpos_HLA4',
                                      description='alpha = 0.1')
    trainable_models.append(trainable_1)
    trainable_1.train_Att()














