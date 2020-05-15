import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings('ignore')
import torchvision.models as models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,roc_auc_score
import pickle
from metric import print_f_score

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomAffine(0, shear=0.2, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def Load_Image_Information(path):
    image_Root_Dir = './image_Merge/'
    iamge_Dir = os.path.join(image_Root_Dir, path)
    return Image.open(iamge_Dir).convert('RGB')

class my_Data_Set(nn.Module):
    def __init__(self, meta_filepath, transform=None, target_transform=None, loader=None):
        super(my_Data_Set, self).__init__()
        metainfo = pd.read_csv(meta_filepath)
        images = []
        symptom = []
        for i in range(metainfo.shape[0]):
            images.append(metainfo.iloc[i, 0])
            symptom.append(metainfo.iloc[i, 1:].values.tolist())
        mlb = MultiLabelBinarizer(classes=['CVD19', 'GGO', 'Csld', 'CrPa', 'Aibr', 'InSep'])
        labels = np.array(mlb.fit_transform(symptom), dtype=np.float64)
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, item):
        imageName = self.images[item]
        label = self.labels[item]
        image = self.loader(imageName)
        if self.transform is not None:
            image = self.transform(image)
        label = torch.FloatTensor(label)
        return image, label

    def __len__(self):
        return len(self.images)


class densenet121_COVID(nn.Module):
    def __init__(self):
        super(densenet121_COVID,self).__init__()
        net = models.densenet121(pretrained=True)
        num_input = net.classifier.in_features
        net.classifier = nn.Linear(num_input, 6)
        self.densenet121_out = net
    def forward(self,x):
        x=self.densenet121_out(x)
        # print(x.shape)
        return F.sigmoid(x)

class loss_fun(nn.Module):
    def __init__(self):
        super(loss_fun, self).__init__()

    def forward(self, output, target):
        loss = torch.zeros((target.shape[0], target.shape[1])).to(device)
        loss1 = self.__loss_nosym(output, target).to(device)
        loss2 = self.__loss_sym(output, target).to(device)
        for i in range(target.shape[0]):
            w1 = torch.prod((target[i][1:] > 0).any()).to(device)
            w2 = (1 - w1).to(device)
            loss[i] = w2 * loss1[i] + w1 * loss2[i]
        loss[:,0] = 5 * loss[:,0]
        return torch.mean(loss),torch.mean(loss1)

    def __loss_nosym(self, output, target):
        loss = torch.zeros((target.shape[0], target.shape[1]))
        loss[:, 0] = criterion(output, target)[:, 0]
        return loss

    def __loss_sym(self, output, target):
        loss = criterion(output, target)
        return loss
def test(model, model_path,test_loader):

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    test_loss = 0
    correct = 0
    predicates_all, target_all = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #print(output)
            _,covid_loss=loss_fun(output, target)
            test_loss += (covid_loss*data.shape[0]).item()
            pred = output.ge(0.5).float()[:,0]
            correct += pred.eq(target[:,0].data.view_as(pred)).sum()
            predicates_all += pred.cpu().numpy().tolist()
            target_all += target[:,0].data.cpu().numpy().tolist()
    test_loss /= len(test_loader.dataset)
    correct=correct.item()
    test_acc=100. * correct / len(test_loader.dataset)
    print('\nTest-->  Avg. loss: {:.6f}  acc: {:.3f}% '.format(test_loss,test_acc))
    return predicates_all, target_all

if __name__ == '__main__':
    LR = 0.00005

    test_dataset = my_Data_Set('./data_split/test_meta.csv', transform=data_transforms['val'],
                               loader=Load_Image_Information)
    if len(test_dataset) < 4:
        val_test_batchsize = 1
    else:
        val_test_batchsize = 4

    test_loader = DataLoader(test_dataset, val_test_batchsize, shuffle=False)

    cnn = densenet121_COVID()
    print(cnn)
    device = torch.device("cuda:14" if torch.cuda.is_available() else "cpu")
    cnn.to(device)

    optimizer = optim.Adam(cnn.parameters(), lr=LR, weight_decay=0.0005)
    criterion = nn.BCELoss(size_average=False, reduce=False)
    loss_fun = loss_fun()

    model_path = './best_model/COVID19_densenet121_94_97.path.tar'
    predicates_all, target_all = test(cnn, model_path, test_loader)
    print(predicates_all)
    print(target_all)
    print_f_score(predicates_all, target_all)
    print(accuracy_score(target_all, predicates_all))
    print(classification_report(target_all, predicates_all))
    print("AUC",roc_auc_score(target_all, predicates_all))
    print(confusion_matrix(target_all, predicates_all))