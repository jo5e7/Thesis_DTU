import sys
sys.path.append('../../')
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from PADChest_DataLoading import PadChestDataset_loc, Resize_loc, RandomRotation_loc, ToTensor_loc, Normalize_loc, RandomHorizontalFlip_loc
from torchvision import transforms
from torchvision.transforms import Resize, RandomRotation, ToTensor, Normalize, RandomHorizontalFlip
import torch.nn.functional as F
from torchvision.transforms import functional as FVision
import cv2
from LR.With_HM.DenseNet_LOC_HM import DenseNet_LOC_HM, loadSD_densenet_loc_hm_169

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


parser = argparse.ArgumentParser()
parser.add_argument('--root')
parser.add_argument('--model_url')
args = parser.parse_args()

root_folder = "D:\PADChest\images8"
model_name = ''
if vars(args)['root'] is not None:
    root_folder = vars(args)['root']
    model_name = vars(args)['model_name']


csv_hm = "PADChest_hm_LRUMDP_opacity.csv"


model_url = model_name + '/' + model_name + '.pth'
save_path = model_name + '/'
batch_size = 1


radiographic_findings_opacity = ['opacity']
locations_labels = ['loc left', 'loc right', 'loc upper', 'loc middle', 'loc lower', 'loc pleural', 'loc mediastinum']

transforms_test_loc = transforms.Compose(
        [Resize_loc(512), ToTensor_loc(), Normalize_loc(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
hm_dataset = PadChestDataset_loc(csv_hm, radiographic_findings_opacity, locations_labels, root_folder, transform=transforms_test_loc)
hm_loader = torch.utils.data.DataLoader(hm_dataset, batch_size=1, num_workers=1)



unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

for enum, data in enumerate(hm_loader, 0):
    #print(enum)
    images, labels, _ = data
    image_hm = images
    images = images.cuda()
    labels = labels.cuda()
    # net = DenseNet_MH()
    # net.load_state_dict(torch.load(self.name + '/' + self.name + '.pth'))
    sd = torch.load(model_url)
    net = loadSD_densenet_loc_hm_169(model_state_dict=sd)
    net.eval()

    pred, _ = net(images)
    pred = torch.sigmoid(pred)
    # get the gradient of the output with respect to the parameters of the model


    #print(pred)
    print(pred[:, 0])
    pred[:, 0].backward()
    #print(pred)
    #print(pred.shape)

    # pull the gradients out of the model
    gradients = net.module.get_activations_gradient()
    #print('gradients', gradients.shape)

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    #print('pooled_gradients', pooled_gradients.shape)

    # get the activations of the last convolutional layer
    activations = net.module.get_activations(images).detach()
    #print('activations', activations.shape)

    # weight the channels by corresponding gradients
    #print('gradients.shape[1]', gradients.shape[1])
    for i in range(gradients.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    #print('heatmap', heatmap.shape)
    # heatmap = heatmap.cpu().numpy()

    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    # heatmap = np.maximum(heatmap, 0)
    heatmap = F.relu(heatmap)
    #print('heatmap_relu', heatmap.shape)
    #print('heatmap_relu', heatmap)

    # normalize the heatmap
    # print(torch.from_numpy(heatmap).float())
    heatmap /= torch.max(heatmap)
    #print('heatmap_/=', heatmap.shape)
    #print('heatmap_/=', heatmap)


    # Print class
    #print('class', labels.cpu().numpy())

    # draw the heatmap
    heatmap = heatmap.cpu().numpy()
    #print('heatmap_numpy', heatmap.shape)
    #print('heatmap_numpy', heatmap)
    plt.matshow(heatmap.squeeze())
    plt.show()
    # plt.savefig(str(enum)+'.png', bbox_inches='tight')

    # heatmanp on image

    # img = cv2.imread('./data/Elephant/data/05fig34.jpg')
    #print(image_hm.shape)
    image_hm = image_hm.view(image_hm.shape[1], image_hm.shape[2], image_hm.shape[3])
    #print(image_hm.shape)
    unorm(image_hm)
    img = FVision.to_pil_image(image_hm, 'RGB')
    # print(img.shape)
    img = np.array(img)
    #print(img.shape)
    img = img[:, :, ::-1].copy()
    #print(img.shape)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    #print('heatmap-shape', heatmap.shape)
    #print('img-shape', img.shape)
    superimposed_img = heatmap * 0.4 + img
    # cv2.imwrite(save_path + str(enum) + '_img.png', img)
    # cv2.imwrite(save_path + str(enum) + '_heatmap.png', heatmap)
    cv2.imwrite(save_path + str(enum) + '.png', superimposed_img)
    pass