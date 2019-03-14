import torch
import torchvision
from torch import nn, optim

def run_ResN(epochs=10, resize=224, batch_size=20, workers=2):
    train_data_loader, test_data_loader, validation_data_loader, in_features, num_classes, classes = get_dataloaders(
           'training',
            'test',
            'validation', batch_size, rescale=resize, workers=workers)

    net = torchvision.models.densenet121(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, num_classes)
    net = net.cuda()

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters())
    criterion = nn.BCEWithLogitsLoss()


    def get_model_accuracy(data_set, model_name):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in data_set:
                images, labels = data
                images = images.cuda()
                labels = labels.cuda()
                # print("test images shape", images.shape)

                outputs = net(images)


                #_, predicted = torch.max(outputs.data, 1)
                total += labels.size(0) * labels.size(1)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the ' + model_name + ' images: %d %%' % (
                100 * correct / total))
        return (100 * correct / total)

    final_net_accuracy = 0
    for epoch in range(epochs):  # loop over the data-set multiple times
        net.train()
        running_loss = 0.0
        #if optim_steper is not None:
        #    scheduler.step()

        for i, data in enumerate(train_data_loader, 0):
            # get the inputs
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 400 == 399:  # print every 50 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 400))
                running_loss = 0.0

        net.eval()
        temp_net_accuracy = get_model_accuracy(test_data_loader, 'test') + get_model_accuracy(validation_data_loader, 'val')
        temp_net_accuracy = temp_net_accuracy/2
        if temp_net_accuracy > final_net_accuracy:
            final_net_accuracy = temp_net_accuracy
            torch.save(net, 'ResNet18_{}_{}.pth'.format(epoch, round(temp_net_accuracy,3)))
            torch.save(net, 'ResNet18.pth')


    net = torch.load('ResNet18.pth')
    net.eval()

    get_model_accuracy(test_data_loader, 'test')
    get_model_accuracy(validation_data_loader, 'val')
    get_model_accuracy(train_data_loader, 'train')

    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    with torch.no_grad():
        for data in test_data_loader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(num_classes):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    pass



if __name__ == '__main__':
    #run_ResN(workers=5, epochs=20)

    model = torch.load("ResNet18_No_Norm_11_97.729.pth", map_location='cpu')
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, "ResNet18_No_Norm_11_97.729.onnx", verbose=True)