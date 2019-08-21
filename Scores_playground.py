import torch
import Prediction_scores
import numpy as np
from PADChest_DataLoading import PadChestDataset_loc_with_inputs, Resize_loc, RandomRotation_loc, ToTensor_loc, Normalize_loc, RandomHorizontalFlip_loc
from torchvision import transforms

labels_names = ['nodule', 'multiple nodules', 'infiltrates', 'interstitial pattern', 'ground glass pattern', 'reticular interstitial pattern', 'reticulonodular interstitial pattern', 'miliary opacities', 'alveolar pattern', 'consolidation', 'air bronchogram', 'air bronchogram', 'atelectasis', 'total atelectasis', 'lobar atelectasis', 'segmental atelectasis', 'laminar atelectasis', 'round atelectasis', 'atelectasis basal', 'mass', 'mediastinal mass', 'breast mass', 'pleural mass', 'pulmonary mass', 'soft tissue mass', 'pneumonia', 'atypical pneumonia', 'pulmonary edema']
#labels_names = ['calcified densities', 'nodule', 'infiltrates', 'atelectasis', 'pleural thickening', 'pleural effusion', 'hilar enlargement', 'aortic elongation', 'mediastinal enlargement', 'mass', 'thoracic cage deformation', 'vertebral degenerative changes', 'fracture']
csv_test = r"C:\Users\maest\OneDrive\DTU\Semestre 4\Thesis\Code\CheXNet_aproach\Datase_stratification\PADChest_val_LRUMDP_opacity.csv"
root_folder = "D:\PADChest\images8"
radiographic_findings_opacity = ['opacity']
locations_labels = ['loc left', 'loc right', 'loc upper', 'loc middle', 'loc lower', 'loc pleural', 'loc mediastinum']


transforms_test_loc = transforms.Compose([Resize_loc(512), ToTensor_loc(),
                                          Normalize_loc(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
test_dataset = PadChestDataset_loc_with_inputs(csv_test, radiographic_findings_opacity, locations_labels, root_folder,
                                   transform=transforms_test_loc, testing=False, pos_labels_always=True,
                                    original_labels=labels_names)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=4,
                                                   drop_last=True)




def get_model_accuracy(model, data_loader):
    all_y = []
    all_y_pred = []
    all_opacity_labels = []
    with torch.no_grad():
        for data in data_loader:
            images, labels, opacity_labels = data
            images = images.cuda()
            labels = labels.cuda()
            opacity_labels = opacity_labels.cuda()
            # print("test images shape", images.shape)

            outputs, _ = model(images, True)
            outputs = torch.sigmoid(outputs)
            # print('labels:', labels)
            # print('outputs:', outputs)

            labels = labels.cpu().numpy()
            outputs = outputs.cpu().numpy()
            opacity_labels = opacity_labels.cpu().numpy()

            for a in labels:
                all_y.append(a)

            for a in outputs:
                all_y_pred.append(a)

            for l in opacity_labels:
                all_opacity_labels.append(l)


        results = Prediction_scores.get_micro_roc_auc_score_for_all_opacity(all_y, all_y_pred,all_opacity_labels, labels_names)
        return results

if __name__ == '__main__':
    # LR Model
    from LR.With_HM.DenseNet_LOC_HM import DenseNet_LOC_HM, loadSD_densenet_loc_hm_169
    from Attention.With_HM.DenseNet_Att_HM import loadSD_densenet_att_hm_169
    from Attention_QKV_Avg_Poll.DenseNet_AttQKV_HM import loadSD_densenet_attQKV_2_hm_169

    basic_loc_url = r'D:\PADChest\Final_models\2_BCE_SGD0.005_a1_MLr_AlwaysPos_BCEPos.pth'
    guide_att_url = r'D:\PADChest\Final_models\2_BCE_SGD0.001_a1_MLr_AlwaysPos_BCEPos_BPpos_HiddenL1.pth'
    transformer_abg_pool = r'D:\PADChest\Final_models\2_BCE_SGD0.001_a1_MAr_BPpos_d832_Hn4_AvgPool.pth'
    sd = torch.load(transformer_abg_pool)
    #net = loadSD_densenet_loc_hm_169(model_state_dict=sd)
    #net = loadSD_densenet_att_hm_169(model_state_dict=sd)
    net = loadSD_densenet_attQKV_2_hm_169(model_state_dict=sd, dq=832, dv=832)
    net.eval()
    results = get_model_accuracy(net, test_dataloader)
    print(results)


    import matplotlib.pyplot as plt
    import operator
    import collections



    results_sorted = sorted(results.items(), key=operator.itemgetter(1))
    results_sorted.reverse()
    results_sorted = collections.OrderedDict(results_sorted)
    keys_list = list(results_sorted.keys())
    auc_list = []
    for k in keys_list:
        auc_list.append(results_sorted[k])


    p1 = plt.bar(np.arange(len(keys_list)), auc_list)
    plt.ylabel('AUC Scores')
    plt.xlabel('label')
    plt.title('AUC scores by diagnosis (Transformer att)')
    plt.xticks(np.arange(len(keys_list)), keys_list)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()