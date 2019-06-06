import torch
import torchvision
import Prediction_scores
import numpy as np
import os
import cv2
from Baseline.With_HM.DenseNet_HM import DenseNet_MH, loadSD_densenet_hm_169

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

class TrainableModel:

    def __init__(self, model, optimizer, loss_criterion, train_loader, val_loader,
                 test_loader=None, hm_loader=None, name=None, description='', score_type='macro_roc_auc'):

        self.model = model
        self.optimizer = optimizer
        self.loss_criterion = loss_criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.score_type = score_type
        self.description = description
        self.hm_loader = hm_loader

        if test_loader is not None:
            self.test_loader = test_loader

        if name is not None:
            self.name = name
        else:
            self.name = ''


    def set_name(self, name):
        self.name = name

    def log(self, str):
        with open(self.name + '/' + self.name + '_log' + '.txt', "a") as text_file:
            print(str, file=text_file)

    def get_model_accuracy(self, model, data_loader, data_name, score_type, save_samples=False):
        all_y = []
        all_y_pred = []
        with torch.no_grad():
            for data in data_loader:
                images, labels = data
                images = images.cuda()
                labels = labels.cuda()
                # print("test images shape", images.shape)

                outputs = model(images, True)
                outputs = torch.sigmoid(outputs)
                # print('labels:', labels)
                # print('outputs:', outputs)

                labels = labels.cpu().numpy()
                outputs = outputs.cpu().numpy()

                for a in labels:
                    all_y.append(a)

                for a in outputs:
                    all_y_pred.append(a)

        # print(all_y)
        # print(all_y_pred)

        if save_samples:
            self.log('micro_F1: ' + str(Prediction_scores.get_micro_f1_score(all_y, all_y_pred)))
            self.log('macro_F1 per class: ' + str(Prediction_scores.get_macro_f1_score(all_y, all_y_pred)))
            self.log('macro_F1: ' + str(np.average(Prediction_scores.get_macro_f1_score(all_y, all_y_pred))))
            self.log('micro_roc_auc: ' + str(Prediction_scores.get_micro_roc_auc_score(all_y, all_y_pred)))
            self.log('macro_roc_auc per class: ' + str(Prediction_scores.get_macro_roc_auc_score(all_y, all_y_pred)))
            self.log('macro_roc_auc: ' + str(np.average(Prediction_scores.get_macro_roc_auc_score(all_y, all_y_pred))))
            self.log(all_y)
            self.log(all_y_pred)

        if score_type == 'micro_F1':
            f1_score = Prediction_scores.get_micro_f1_score(all_y, all_y_pred)
            print(self.name, data_name, 'F1:', f1_score)
            return f1_score
        elif score_type == 'macro_F1':
            f1_score = Prediction_scores.get_macro_f1_score(all_y, all_y_pred)
            print(self.name, data_name, 'F1 per class:', f1_score)
            print(self.name, data_name, 'F1:', np.average(f1_score))
            return np.average(f1_score)
        elif score_type == 'micro_roc_auc':
            score = Prediction_scores.get_micro_roc_auc_score(all_y, all_y_pred)
            print(self.name, data_name, 'ROC_AUC_score:', score)
            return score
        elif score_type == 'macro_roc_auc':
            score = Prediction_scores.get_macro_roc_auc_score(all_y, all_y_pred)
            print(self.name, data_name, 'ROC_AUC_score per clase:', score)
            print(self.name, data_name, 'ROC_AUC_score:', np.average(score))
            return np.average(score)

    def train(self, epochs=100, tolerance=2):

        net = self.model
        optimizer = self.optimizer
        criterion = self.loss_criterion

        epoch = 0

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=tolerance - 1)

        if epochs is None:
            epochs = -1

        epochs_without_imporving = 0
        final_net_accuracy = 0
        epoch_losses_list = []
        accuracies_list = []
        decays = 0

        os.mkdir(self.name)
        self.log(self.name)
        self.log(self.description)
        self.log('Train criteria: ' + self.score_type)

        while (epoch != epochs) & (epochs_without_imporving <= tolerance):
            # self.log('********************************')
            self.log('************' + 'Epoch: ' + str(epoch) + '************')
            # self.log('********************************')
            print('Epoch:', epoch)
            running_loss = 0.0
            epoch_losses = []

            # Learning rate decay if accuracy is not improving
            if (epochs_without_imporving == 2) & (decays < 2):
                decays += 1
                #net = DenseNet_MH()
                #net.load_state_dict(torch.load(self.name + '/' + self.name + '.pth'))
                sd = torch.load(self.name + '/' + self.name + '.pth')
                net = loadSD_densenet_hm_169(model_url=sd)
                epochs_without_imporving = 0
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] / 10
                    print('decays', decays)
                    print('Learning rate', g['lr'])
                    self.log('Learning rate changed to: ' + str(g['lr']))

            net.train()
            for i, data in enumerate(self.train_loader, 0):
                # get the inputs
                inputs, labels = data
                # print(labels.shape)
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize

                outputs = net(inputs)
                # print(labels.shape)
                # print(outputs.shape)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                epoch_losses.append(loss.item())
                if i % 100 == 99:  # print every 50 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0

            net.eval()
            temp_net_accuracy = self.get_model_accuracy(net, self.val_loader, 'val', self.score_type)
            accuracies_list.append(temp_net_accuracy)
            epoch_losses_list.append(np.average(np.array(epoch_losses)))

            self.log('Score : ' + str(temp_net_accuracy))
            self.log('Loss : ' + str(np.average(np.array(epoch_losses))))

            if temp_net_accuracy > final_net_accuracy:
                final_net_accuracy = temp_net_accuracy
                torch.save(net.state_dict(), self.name + '/' + self.name + '_{}_{}.pth'.format(epoch, round(temp_net_accuracy, 2)))
                torch.save(net.state_dict(), self.name + '/' + self.name + '.pth')
                epochs_without_imporving = 0
            else:
                epochs_without_imporving += 1

            epoch += 1

        self.log('Scores list per epoch:')
        self.log(str(accuracies_list))
        self.log('Losses list per epoch:')
        self.log(str(epoch_losses_list))
        if self.test_loader is not None:
            # net = DenseNet_MH()
            sd = torch.load(self.name + '/' + self.name + '.pth')
            net = loadSD_densenet_hm_169(model_url=sd)
            net.eval()
            self.log('Final Scores for test set')
            final_score = self.get_model_accuracy(net, self.test_loader, 'test', self.score_type, save_samples=True)
            self.log('Final Score for test set: ' + str(final_score))

            # Heatmap creation

            import matplotlib.pyplot as plt
            import torch.nn.functional as F
            from torchvision.transforms import functional as FVision
            unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


            imagelist = []

            for enum, data in enumerate(self.hm_loader, 0):
                print(enum)
                images, labels = data
                image_hm = images
                images = images.cuda()
                labels = labels.cuda()
                #net = DenseNet_MH()
                sd = torch.load(self.name + '/' + self.name + '.pth')
                net = loadSD_densenet_hm_169(model_url=sd)
                net.eval()

                pred = net(images)
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
                plt.matshow(heatmap.squeeze())
                plt.show()
                #plt.savefig(str(enum)+'.png', bbox_inches='tight')

                # heatmanp on image

                #img = cv2.imread('./data/Elephant/data/05fig34.jpg')
                print(image_hm.shape)
                image_hm = image_hm.view(image_hm.shape[1], image_hm.shape[2], image_hm.shape[3])
                print(image_hm.shape)
                unorm(image_hm)
                img = FVision.to_pil_image(image_hm, 'RGB')
                #print(img.shape)
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
                cv2.imwrite(str(enum) + '_img.png', img)
                cv2.imwrite(str(enum) + '_heatmap.png', heatmap)
                cv2.imwrite(str(enum)+'.png', superimposed_img)
                pass


