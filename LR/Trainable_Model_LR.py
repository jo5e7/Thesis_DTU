import torch
import os
import numpy as np
import Prediction_scores


class Trainable_Model_LR:

    def __init__(self, model, optimizer, loss_criterion_1, loss_criterion_2, train_loader, val_loader,
                 test_loader=None, name=None, description='', score_type='macro_roc_auc'):

        self.model = model
        self.optimizer = optimizer
        self.loss_criterion_1 = loss_criterion_1
        self.loss_criterion_2 = loss_criterion_2
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.score_type = score_type
        self.description = description

        if test_loader is not None:
            self.test_loader = test_loader

        if name is not None:
            self.name = name
        else:
            self.name = ''

    def log(self, str):
        with open(self.name + '/' + self.name + '_log' + '.txt', "a") as text_file:
            print(str, file=text_file)

    def get_model_accuracy(self, model, data_loader, data_name, score_type, save_samples=False):
        all_y = []
        all_y_pred = []
        all_locs = []
        all_locs_pred = []
        with torch.no_grad():
            for data in data_loader:
                images, labels, locs = data
                images = images.cuda()
                labels = labels.cuda()
                locs = locs.cuda()
                # print("test images shape", images.shape)

                outputs, outputs_loc = model(images)
                outputs = torch.sigmoid(outputs)
                outputs_loc = torch.sigmoid(outputs_loc)
                # print('labels:', labels)
                # print('outputs:', outputs)

                labels = labels.cpu().numpy()
                outputs = outputs.cpu().numpy()
                locs = locs.cpu().numpy()
                outputs_loc = outputs_loc.cpu().numpy()

                for a in labels:
                    all_y.append(a)

                for a in outputs:
                    all_y_pred.append(a)

                for l in locs:
                    all_locs.append(l)

                for l in outputs_loc:
                    all_locs_pred.append(l)

        # print(all_y)
        # print(all_y_pred)

        if save_samples:
            self.log('micro_F1: ' + str(Prediction_scores.get_micro_f1_score(all_y, all_y_pred)))
            self.log('macro_F1 per class: ' + str(Prediction_scores.get_macro_f1_score(all_y, all_y_pred)))
            self.log('macro_F1: ' + str(np.average(Prediction_scores.get_macro_f1_score(all_y, all_y_pred))))
            self.log('micro_roc_auc: ' + str(Prediction_scores.get_micro_roc_auc_score(all_y, all_y_pred)))
            self.log('macro_roc_auc per class: ' + str(Prediction_scores.get_macro_roc_auc_score(all_y, all_y_pred)))
            self.log('macro_roc_auc: ' + str(np.average(Prediction_scores.get_macro_roc_auc_score(all_y, all_y_pred))))
            presicion, recall, thresholds = Prediction_scores.get_precision_recall_curve(all_y, all_y_pred)
            self.log('presicion class: ' + str(presicion))
            self.log('recall class: ' + str(recall))
            self.log('threshold class: ' + str(thresholds))
            # self.log(all_y)
            # self.log(all_y_pred)
            self.log("________________________________")
            self.log('macro_F1 per loc: ' + str(Prediction_scores.get_macro_f1_score(all_locs, all_locs_pred)))
            self.log('macro_F1: ' + str(np.average(Prediction_scores.get_macro_f1_score(all_locs, all_locs_pred))))
            self.log('macro_roc_auc per loc: ' + str(Prediction_scores.get_macro_roc_auc_score(all_locs, all_locs_pred)))
            self.log('macro_roc_auc: ' + str(np.average(Prediction_scores.get_macro_roc_auc_score(all_locs, all_locs_pred))))
            #presicion, recall, thresholds = Prediction_scores.get_precision_recall_curve(all_locs, all_locs_pred)
            #self.log('presicion loc: ' + str(presicion))
            #self.log('recall loc: ' + str(recall))
            #self.log('threshold loc: ' + str(thresholds))
            #self.log("")

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
            score_loc = Prediction_scores.get_macro_roc_auc_score(all_locs, all_locs_pred)
            #print('All y:')
            #print(all_y)
            #print('All y predictions:')
            #print(all_y_pred)
            #print('All locs:')
            #print(all_locs)
            #print('All locs predictions:')
            #print(all_locs_pred)
            print(self.name, data_name, 'ROC_AUC_score per clase:', score)
            print(self.name, data_name, 'ROC_AUC_score:', np.average(score))
            print(self.name, data_name, 'ROC_AUC_score per clase:', score_loc)
            print(self.name, data_name, 'ROC_AUC_score:', np.average(score_loc))


            #self.log('presicion class: ' + str(presicion))
            #self.log('recall class: ' + str(recall))
            #self.log('threshold class: ' + str(thresholds))
            return np.average(score)

    def train_LR(self, epochs=100, tolerance=2):
        net = self.model
        optimizer = self.optimizer
        criterion = self.loss_criterion_1
        criterion_non_zero = self.loss_criterion_2

        epoch = 0

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=tolerance - 1)

        if epochs is None:
            epochs = -1

        epochs_without_imporving = 0
        final_net_accuracy = 0
        epoch_losses_list = []
        class_epoch_losses_list = []
        loc_epoch_losses_list = []
        accuracies_list = []
        decays = 0

        os.mkdir(self.name)
        self.log(self.name)
        self.log(self.description)
        self.log('Train criteria: ' + self.score_type)
        pass

        while (epoch != epochs) & (epochs_without_imporving <= tolerance):
            # self.log('********************************')
            self.log('************' + 'Epoch: ' + str(epoch) + '************')
            # self.log('********************************')
            print('Epoch:', epoch)
            net.train()
            total_running_loss = 0.0
            class_running_loss = 0.0
            loc_running_loss = 0.0
            epoch_losses = []
            class_epoch_losses = []
            loc_epoch_losses = []

            # Learning rate decay if accuracy is not improving
            if (epochs_without_imporving == 2) & (decays < 2):
                decays += 1
                net = torch.load(self.name + '/' + self.name + '.pth')
                epochs_without_imporving = 0
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] / 10
                    print('decays', decays)
                    print('Learning rate', g['lr'])
                    self.log('Learning rate changed to: ' + str(g['lr']))

            for i, data in enumerate(self.train_loader, 0):
                # get the inputs
                inputs, labels, locations = data
                # print(labels.shape)
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    locations = locations.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize

                outputs_class, outputs_locations = net(inputs)
                # print(labels.shape)
                # print(outputs.shape)
                class_loss = criterion(outputs_class, labels)
                loc_loss = criterion_non_zero(outputs_locations, locations)
                loss = class_loss + loc_loss
                loss.backward()
                optimizer.step()
                # print('L', loss)
                # print('CL', class_loss)
                # print('LL', loc_loss)

                # print statistics
                total_running_loss += loss.item()
                class_running_loss += class_loss.item()
                loc_running_loss += loc_loss.item()
                epoch_losses.append(loss.item())
                class_epoch_losses.append(class_loss.item())
                loc_epoch_losses.append(loc_loss.item())
                if i % 100 == 99:  # print every 500 mini-batches
                    print('[%d, %5d] Total loss: %.3f' %
                          (epoch, i + 1, total_running_loss / 100), end=' | ')
                    print('Classification loss: %.3f' %
                          (class_running_loss / 100), end=' | ')
                    print('Localization loss: %.3f' %
                          (loc_running_loss / 100))
                    #print(class_running_loss)
                    total_running_loss = 0.0
                    class_running_loss = 0.0
                    loc_running_loss = 0.0

            net.eval()
            temp_net_accuracy = self.get_model_accuracy(net, self.val_loader, 'val', self.score_type)
            accuracies_list.append(temp_net_accuracy)
            epoch_losses_list.append(np.average(np.array(epoch_losses)))
            class_epoch_losses_list.append(np.average(np.array(class_epoch_losses)))
            loc_epoch_losses_list.append(np.average(np.array(loc_epoch_losses)))

            self.log('Score : ' + str(temp_net_accuracy))
            self.log('Total Loss : ' + str(np.average(np.array(epoch_losses))))
            self.log('Classification Loss : ' + str(np.average(np.array(class_epoch_losses))))
            self.log('Localization Loss : ' + str(np.average(np.array(loc_epoch_losses))))

            if temp_net_accuracy > final_net_accuracy:
                final_net_accuracy = temp_net_accuracy
                torch.save(net, self.name + '/' + self.name + '_{}_{}.pth'.format(epoch, round(temp_net_accuracy, 2)))
                torch.save(net, self.name + '/' + self.name + '.pth')
                epochs_without_imporving = 0
            else:
                epochs_without_imporving += 1

            epoch += 1

        self.log('Scores list per epoch:')
        self.log(str(accuracies_list))
        self.log('Losses list per epoch:')
        self.log(str(epoch_losses_list))
        self.log('Classification losses list per epoch:')
        self.log(str(class_epoch_losses_list))
        self.log('Localization losses list per epoch:')
        self.log(str(loc_epoch_losses_list))
        if self.test_loader is not None:
            net = torch.load(self.name + '/' + self.name + '.pth')
            self.log('Final Scores for test set')
            final_score = self.get_model_accuracy(net, self.test_loader, 'test', self.score_type, save_samples=True)
            self.log('Final Score for test set: ' + str(final_score))


    pass