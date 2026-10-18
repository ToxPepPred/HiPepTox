import torch
import torch.nn as nn
import copy
import random
import warnings
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import BertTokenizer, AutoTokenizer, BertModel, AutoModel
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score,precision_recall_curve, average_precision_score
from sklearn.model_selection import KFold, RepeatedKFold
warnings.filterwarnings('ignore')

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
random_seed(777)

device = torch.device("cuda:0")

train_data_path = pd.read_csv("train_set7_label.csv")

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.6, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
def tokenize(path):
    data_frame = path
    data_columns = data_frame.columns.tolist()
    data_columns = [i for i in data_columns]
    data_frame.columns = data_columns
    trainlabel = data_frame[data_frame.columns[0]]
    proBert_seq = data_frame[data_frame.columns[1]]
    return  np.array(trainlabel), np.array(proBert_seq)

train_label, train_X = tokenize(train_data_path)

def Model_Evaluate(confus_matrix):
    TN, FP, FN, TP = confus_matrix.ravel()
    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    MCC = ((TP * TN) - (FP * FN)) / (np.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)))
    Pre = TP / (TP + FP)
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)

    # 计算BACC
    BACC = (TPR + TNR) / 2
    return SN, SP, ACC, MCC,BACC, Pre

def cal_score(pred, label):
    try:
        AUC = roc_auc_score(list(label), pred)

    except:
        AUC = 0
    pred = np.argmax(pred, axis=1)
    label = np.array(label)
    cm = confusion_matrix(label, pred, labels=None, sample_weight=None)
    SN, SP, ACC, MCC,BACC, Pre = Model_Evaluate(cm )
    print(
        "Model score --- SN:{0:.3f}       SP:{1:.3f}       ACC:{2:.3f}       MCC:{3:.3f}   BACC:{4:.3f}    Pre:{4:.3f}   AUC:{5:.3f}".format(
            SN, SP, ACC, MCC,BACC,Pre, AUC))

    return BACC,cm


class AQYDataset(Dataset):
    def __init__(self, df, label, device):
        self.protein_seq = df
        self.label_list = label

    def __getitem__(self, index):
        seq = self.protein_seq[index]
        seq = seq.replace('', ' ')
        encoding = tokenizer.encode_plus(
            seq,
            add_special_tokens=True,
            max_length=50,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )

        sample = {
            'input_ids': encoding['input_ids'].flatten(),
            # 'token_type_ids': encoding['token_type_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),

        }
        label = self.label_list[index]
        return sample,  label

    def __len__(self):
        return len(self.protein_seq)


def fit(model, train_loader, optimizer,  device):
    model.train()
    outlist = []
    pred_list = []
    label_list = []

    accumulation_steps = 2
    for i, (samples,  label) in enumerate(train_loader):
        input_ids = samples['input_ids'].to(device)
        # token_type_ids = samples['token_type_ids'].to(device)
        attention_mask = samples['attention_mask'].to(device)
        label = torch.tensor(label).to(device)
        pred = model(input_ids, attention_mask)
        pred = pred.squeeze()
        loss2 = criterion(pred, label)
        loss = loss2
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pred_list.extend(pred.squeeze().cpu().detach().numpy())

        label_list.extend(label.squeeze().cpu().detach().numpy())
    pred_list = np.array(pred_list)
    label_list = np.array(label_list)
    score,_ = cal_score(pred_list, label_list)
    return score,outlist


def validate(model, val_loader, device):
    model.eval()
    pred_list = []
    label_list = []

    for samples,  label in val_loader:
        input_ids = samples['input_ids'].to(device)
        # token_type_ids = samples['token_type_ids'].to(device)
        attention_mask = samples['attention_mask'].to(device)
        label = torch.tensor(label).to(device)
        pred= model(input_ids, attention_mask)
        # pred_list.extend(pred.squeeze().cpu().detach().numpy())
        pred_list.extend(pred.squeeze().cpu().detach().numpy())
        label_list.extend(label.squeeze().cpu().detach().numpy())
    pred_list = np.array(pred_list)
    label_list = np.array(label_list)
    score,cm = cal_score(pred_list, label_list)

    return score, pred_list, label_list,cm

tokenizer = AutoTokenizer.from_pretrained("/home/hd/SGao/esm2_t30")



class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # print(x.shape)
        query = self.query(x)
        # print(query.shape)
        key = self.key(x)
        # print(key.shape)
        value = self.value(x)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attention = self.softmax(scores)
        output = torch.matmul(attention, value)
        return output

class newmodel(nn.Module):

    def __init__(self, embedding_dim=50, hidden_dim=32, n_layers=1):
        super(newmodel, self).__init__()
        self.bert = AutoModel.from_pretrained("/home/hd/SGao/esm2_t30")
        # self.bert = BertModel.from_pretrained("/home/hd/SGao/probert")
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        out_channle = 32
        self.LSTM = nn.LSTM(640, 320, num_layers=2,
                          bidirectional=True, dropout=0.3)
        self.global_avg_pooling_layer = nn.AdaptiveAvgPool1d(1)
        # self.conv1 = nn.Conv1d(1024, 256, kernel_size=5, stride=1, padding='same')
        # self.conv2 = nn.Conv1d(256, 128, kernel_size=5, stride=1, padding='same')
        # self.conv3 = nn.Conv1d(128, 32, kernel_size=5, stride=1, padding='same')
        self.conv1 = torch.nn.Conv1d(in_channels=1024,
                                     out_channels=256,
                                     kernel_size=2,
                                     stride=1
                                     )
        self.conv2 = torch.nn.Conv1d(in_channels=1024,
                                     out_channels=256,
                                     kernel_size=3,
                                     stride=1
                                     )
        self.conv3 = torch.nn.Conv1d(in_channels=1024,
                                     out_channels=256,
                                     kernel_size=4,
                                     stride=1
                                     )
        self.conv4 = torch.nn.Conv1d(in_channels=1024,
                                     out_channels=256,
                                     kernel_size=5,
                                     stride=1
                                     )

        self.fc1 = nn.Linear(640, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)


        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU()
        self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=5)
        self.dropout = nn.Dropout(0.3)


    def forward(self, input_ids,attention_mask):
        pooled_output,hidden_states = self.bert(
            input_ids=input_ids,
            # token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        y,_= self.LSTM(hidden_states)
        out2 = self.fc1(y)
        out3 = self.fc2(out2)
        out3 = self.relu(out3)
        logit = self.fc3(out3)

        return logit


if __name__ == '__main__':

    X_train = train_X
    label_train = train_label
    skf = KFold(n_splits=5, shuffle=True, random_state=666)
    valid_pred = []
    valid_label = []

    for index, (train_idx, val_idx) in enumerate(skf.split(X_train, label_train)):

        print('**' * 10, '训练-测试开始', '**' * 10)
        proBer_train_seq,proBer_valid_seq  =X_train[train_idx], X_train[val_idx]
        train_Y , valid_Y  =label_train[train_idx], label_train[val_idx]

        train_dataset = AQYDataset(proBer_train_seq, train_Y, device)

        valid_dataset = AQYDataset(proBer_valid_seq, valid_Y, device)
                                            shuffle=True, collate_fn=collate, drop_last=True)

        train_loader = DataLoader(train_dataset,
                                  batch_size=128,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=4)

        valid_loader = DataLoader(valid_dataset,
                                  batch_size=128,
                                  shuffle=False,
                                  drop_last=True,
                                  num_workers=4)

        model = newmodel().to(device)


        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

        criterion = FocalLoss()
        # criterion = nn.CrossEntropyLoss(reduction="sum")

        best_val_score = float('-inf')
        last_improve = 0
        best_model = None

        for epoch in range(40):
            train_score,outlist = fit(model, train_loader, optimizer,  device)
            val_score, _, _,_ = validate(model, valid_loader, device)

            if val_score > best_val_score:
                best_val_score = val_score
                best_model = copy.deepcopy(model)
                last_improve = epoch
                improve = '*'
            else:
                improve = ''

            print(f'Epoch: {epoch + 1} Train Score: {train_score}, Valid Score: {val_score}  ')

        model = best_model
        torch.save(best_model.state_dict(), './num_model{}_acc{}.pkl'.format((index+1 + 1),  best_val_score))
        print(f"=============e0nd!!!!================")
        print("train")
        train_score, _, _,_ = validate(model, train_loader, device)
        print("valid")
        valid_score, pred_list, label_list,cm= validate(model, valid_loader, device)
        valid_pred.extend(pred_list)
        valid_label.extend(label_list)
        print(cm)
print("*****************************************5-fold  valid**********************************************")
print("valid_score")
cross_valid_score,cm = cal_score(valid_pred, valid_label)
print(cm)
