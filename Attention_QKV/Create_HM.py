import sys
sys.path.append('../../')
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from PADChest_DataLoading import PadChestDataset
from torchvision import transforms
from torchvision.transforms import Resize, RandomRotation, ToTensor, Normalize, RandomHorizontalFlip
import torch.nn.functional as F
from torchvision.transforms import functional as FVision
import cv2
from Attention_QKV.DenseNet_AttQKV_HM import DenseNet_att_QKV_HM, loadSD_densenet_attQKV_hm_169

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
parser.add_argument('--hm_csv')
parser.add_argument('--model_url')
args = parser.parse_args()

root_folder = "D:\PADChest\images8"
if vars(args)['root'] is not None:
    root_folder = vars(args)['root']

model_url = r'BCE_SGD0.005_a1_opacity_MLr\BCE_SGD0.005_a1_opacity_MLr.pth'
if vars(args)['model_url'] is not None:
    model_url = vars(args)['model_url']


csv_hm = r"C:\Users\maest\OneDrive\DTU\Semestre 4\Thesis\Code\CheXNet_aproach\Datase_stratification\PADChest_hm_LRUMDP_opacity.csv"
if vars(args)['hm_csv'] is not None:
    csv_hm = vars(args)['hm_csv']

batch_size = 1


radiographic_findings_opacity = ['opacity']
transforms_test = transforms.Compose(
    [Resize(512), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
hm_dataset = PadChestDataset(csv_hm, radiographic_findings_opacity, root_folder, transform=transforms_test)
hm_loader = torch.utils.data.DataLoader(hm_dataset, batch_size=1)



unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

def save_heatmaps(bp_pos, hidden_layers_att, dq, dv,
                                                Att_heads, kernel_att,
                                                stride_att, non_linearity_att, self_att,
                  model_url=model_url, save_path=''):
    for enum, data in enumerate(hm_loader, 0):
        print(enum)
        images, labels = data
        image_hm = images
        images = images.cuda()
        labels = labels.cuda()
        # net = DenseNet_MH()
        # net.load_state_dict(torch.load(self.name + '/' + self.name + '.pth'))
        sd = torch.load(model_url)
        net = loadSD_densenet_attQKV_hm_169(model_state_dict=sd, bp_elementwise=bp_pos, hidden_layers_att=hidden_layers_att,
                                            dq=dq, dv=dv, Att_heads=Att_heads, kernel_att=kernel_att, stride_att=stride_att,
                                            non_linearity_att=non_linearity_att, self_att=self_att
                                         )
        net.eval()

        pred, _ = net(images)
        # get the gradient of the output with respect to the parameters of the model

        print(pred)
        print(pred[:, 0])
        pred[:, 0].backward()
        print(pred)
        print(pred.shape)

        # pull the gradients out of the model
        gradients = net.module.get_activations_gradient()
        print('gradients', gradients.shape)

        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        print('pooled_gradients', pooled_gradients.shape)

        # get the activations of the last convolutional layer
        activations = net.module.get_activations(images).detach()
        print('activations', activations.shape)

        # weight the channels by corresponding gradients
        print('gradients.shape[1]', gradients.shape[1])
        for i in range(gradients.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()
        print('heatmap', heatmap.shape)
        # heatmap = heatmap.cpu().numpy()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        # heatmap = np.maximum(heatmap, 0)
        heatmap = F.relu(heatmap)
        print('heatmap_relu', heatmap.shape)
        print('heatmap_relu', heatmap)

        # normalize the heatmap
        # print(torch.from_numpy(heatmap).float())
        heatmap /= torch.max(heatmap)
        print('heatmap_/=', heatmap.shape)
        print('heatmap_/=', heatmap)

        # Print class
        print('class', labels.cpu().numpy())

        # draw the heatmap
        heatmap = heatmap.cpu().numpy()
        print('heatmap_numpy', heatmap.shape)
        print('heatmap_numpy', heatmap)
        #plt.matshow(heatmap.squeeze())
        #plt.show()
        # plt.savefig(str(enum)+'.png', bbox_inches='tight')

        # heatmanp on image

        # img = cv2.imread('./data/Elephant/data/05fig34.jpg')
        print(image_hm.shape)
        image_hm = image_hm.view(image_hm.shape[1], image_hm.shape[2], image_hm.shape[3])
        print(image_hm.shape)
        unorm(image_hm)
        img = FVision.to_pil_image(image_hm, 'RGB')
        # print(img.shape)
        img = np.array(img)
        print(img.shape)
        img = img[:, :, ::-1].copy()
        print(img.shape)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        print('heatmap-shape', heatmap.shape)
        print('img-shape', img.shape)
        superimposed_img = heatmap * 0.4 + img
        cv2.imwrite(save_path + str(enum) + '_img.png', img)
        cv2.imwrite(save_path + str(enum) + '_heatmap.png', heatmap)
        cv2.imwrite(save_path + str(enum) + '.png', superimposed_img)
        pass