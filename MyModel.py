import os
import numpy as np
import config
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import json
from parser_tool import parameter_parser
from torch.autograd import Variable
from models_update import move_data_to_gpu, BytecodeNet, SBFusionNet
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, roc_curve
# from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import math
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from scipy.stats import ks_2samp
# from torch_geometric.nn import GCNConv
# from transformers import BertModel
plt.switch_backend('Agg')

# torch.cuda.set_device(1)
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
print('using torch', torch.__version__)
args = parameter_parser()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.enabled = False

print(
    args.dataset, args.m, args.lr, args.cuda, args.epoch, args.seed
)


torch.manual_seed(6603)

# class EncoderWeight(nn.Module):
#     def __init__(self, sourcecode_x_train, sourcecode_x_test, bytecode_x_train, bytecode_x_test, train_y,test_y, lr, epochs):
#         super(EncoderWeight, self).__init__()

#         self.sourcecode_x_train = torch.tensor(sourcecode_x_train, dtype=torch.float32)
#         self.sourcecode_x_test = torch.tensor(sourcecode_x_test, dtype=torch.float32)
#         self.bytecode_x_train = torch.tensor(bytecode_x_train, dtype=torch.float32)
#         self.bytecode_x_test = torch.tensor(bytecode_x_test, dtype=torch.float32)
#         self.train_y = torch.tensor(train_y, dtype=torch.float32)
#         self.test_y = torch.tensor(test_y, dtype=torch.float32)
#         self.epochs = epochs
#         # self.class_weight = torch.tensor(compute_class_weight(class_weight='balanced', classes=[0, 1], y=sourcecode_y), dtype=torch.float32)

#         self.graph2vec = nn.Sequential(
#             nn.Linear(1, 250),
#             nn.ReLU(),
#             nn.Linear(250, 200),
#             nn.ReLU()
#         )

#         self.graphweight = nn.Sequential(
#             nn.Linear(200, 1),
#             nn.Sigmoid()
#         )

#         self.pattern1vec = nn.Sequential(
#             nn.Linear(1, 250),
#             nn.ReLU(),
#             nn.Linear(250, 200),
#             nn.ReLU()
#         )

#         # ... Similar blocks for pattern2vec and pattern3vec

#         self.finalmergevec = nn.Sequential(
#             nn.Linear(800, 100),
#             nn.ReLU()
#         )

#         self.prediction = nn.Sequential(
#             nn.Linear(100, 1),
#             nn.Sigmoid()
#         )

#         self.criterion = nn.BCELoss()
#         self.optimizer = optim.Adam(self.parameters(), lr=lr)

#     def forward(self, source_code, byte_code):
#         graph2vec_output = self.graph2vec(graph)
#         graphweight_output = self.graphweight(graph2vec_output)
#         newgraphvec = graph2vec_output * graphweight_output

#         # ... Similar blocks for pattern1vec, pattern2vec, pattern3vec

#         mergevec = torch.cat([newgraphvec, newpattern1vec, newpattern2vec, newpattern3vec], dim=1)
#         flattenvec = mergevec.view(mergevec.size(0), -1)
#         finalmergevec_output = self.finalmergevec(flattenvec)
#         prediction_output = self.prediction(finalmergevec_output)

#         return prediction_output

#     def train_model(self):
#         for epoch in range(self.epochs):
#             self.optimizer.zero_grad()
#             predictions = self.forward(self.sourcecode_x_train, self.bytecode_x_train)
#             loss = self.criterion(predictions, self.train_y.view(-1, 1))
#             loss.backward()
#             self.optimizer.step()

#             print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}')

#     def test_model(self):
#         with torch.no_grad():
#             predictions = self.forward(self.sourcecode_x_test, self.bytecode_x_test)
#             loss = self.criterion(predictions, self.test_y.view(-1, 1))

#             print(f'Test Loss: {loss.item()}')

#             predictions = predictions.round()
#             tn, fp, fn, tp = confusion_matrix(self.y_test.numpy(), predictions.numpy()).ravel()
#             print("Accuracy: ", (tp + tn) / (tp + tn + fp + fn))
#             print('False positive rate(FPR): ', fp / (fp + tn))
#             print('False negative rate(FN): ', fn / (fn + tp))
#             recall = tp / (tp + fn)
#             print('Recall(TPR): ', recall)
#             precision = tp / (tp + fp)
#             print('Precision: ', precision)
#             print('F1 score: ', (2 * precision * recall) / (precision + recall))

class TestClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TestClassifier, self).__init__()
        # self.fc1 = nn.Linear(input_size, hidden_size)
        self.sourcecodeFc = nn.Linear(input_size,128)
        self.bytecodeFc = nn.Linear(hidden_size,128)
        self.relu = nn.ReLU()
        self.attSourcecode = nn.MultiheadAttention(128,num_heads=1)
        self.attBytecode = nn.MultiheadAttention(128,num_heads=1)
        # self.relu = nn.ReLU()
        self.Fc = nn.Linear(256,output_size)
        # self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        # x = x.to(self.fc1_x.weight.device)
        # y = y.to(self.fc1_y.weight.device)
        # z = torch.cat((x,y),dim=1)
        if x.is_sparse:
            x = x.to_dense()
        x = self.sourcecodeFc(x)
        y = self.bytecodeFc(y)
        x = self.relu(x)
        y = self.relu(y)
        x = x.permute(1, 0, 2) 
        x, _ = self.attSourcecode(x, x, x)
        if y.is_sparse:
            y = y.to_dense()
        y = y.permute(1, 0, 2)  # Change shape to (seq_len, batch_size, features) for MultiheadAttention
        y, _ = self.attBytecode(y, y, y)
        y = y.permute(1, 0, 2)
        # x = self.attSourcecode()
        z = torch.cat((x, y), dim=2)
        # x = self.fc1(z)
        # x = self.relu(x)
        # x = self.fc2(x)
        z = self.Fc(z)
        x = self.softmax(z)
        return x
    
# class TestClassifier(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(TestClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x, y):
#         # x = x.to(self.fc1_x.weight.device)
#         # y = y.to(self.fc1_y.weight.device)
#         z = torch.cat((x,y),dim=1)
#         x = self.fc1(z)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.softmax(x)
#         return x
    
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=2,num_heads=2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True,bidirectional=False)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        # LSTM层
        lstm_out, _ = self.lstm(x)
        # ATTention layer
        attn_output, _ = self.self_attention(lstm_out, lstm_out, lstm_out)
        # 取序列的最后一个时间步的输出
        if lstm_out.dim() == 3:
            last_output = attn_output[:, -1, :]
        else:
            last_output = attn_output

        # 全连接层
        fc_out = self.fc(last_output)

        # 应用sigmoid激活函数，得到二分类的输出
        output = self.sigmoid(fc_out)
        
        return output

    # def forward(self, sourcecode,bytecode):
    #     # LSTM层
    #     lstm_out, _ = self.lstm(bytecode)
    #     # ATTention layer
    #     attn_output, _ = self.self_attention(lstm_out, lstm_out, lstm_out)
    #     # 取序列的最后一个时间步的输出
    #     if attn_output.dim() == 3:
    #         last_output = attn_output[:, -1, :]
    #     else:
    #         last_output = attn_output

    #     # 全连接层
    #     fc_out = self.fc(last_output)

    #     # 应用sigmoid激活函数，得到二分类的输出
    #     output = self.sigmoid(fc_out)
        
    #     return output
    
class SimpleCNN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(SimpleCNN, self).__init__()
        
        # 一维卷积层
        self.conv1d = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # 非线性激活函数
        self.relu = nn.ReLU()
        
        # 最大池化层
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc = nn.Linear(32 * 256, output_size)  # 假设输入特征是 64x512

        # 输出层的激活函数
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 一维卷积
        x = self.conv1d(x)
        
        # 非线性激活
        x = self.relu(x)
        
        # 最大池化
        x = self.pool(x)
        
        # 将特征图展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc(x)
        
        # 输出层的激活
        x = self.softmax(x)

        return x



def main():
    workspace = './'
   
    train_sourcecode_path = os.path.join(workspace, 'bytecode','timestamp', 'train_sourcecode1.json')
    train_bytecode_path = os.path.join(workspace, 'bytecode','timestamp' ,'train.json')
    eval_sourcecode_path = os.path.join(workspace, 'bytecode', 'timestamp','test_sourcecode1.json')
    eval_bytecode_path = os.path.join(workspace, 'bytecode', 'timestamp','test.json')
    
    load_sourcecode = open(train_sourcecode_path, 'r')
    train_sourcecode_dict = json.load(load_sourcecode)
    # train_sourcecode_names = np.array([item['contract_name'] for item in sourcecode_dict])
    train_sourcecode_x = np.array([item['node_features'] for item in train_sourcecode_dict])
    train_sourcecode_y = np.array([item['targets'] for item in train_sourcecode_dict])
    load_sourcecode.close()

    load_bytecode = open(train_bytecode_path, 'r')
    train_bytecode_dict = json.load(load_bytecode)
    # train_bytecode_names = np.array([item['contract_name'] for item in bytecode_dict])
    train_bytecode_x = np.array([item['node_features'] for item in train_bytecode_dict])
    # train_bytecode_y = np.array([item['targets'] for item in bytecode_dict])
    load_bytecode.close()

    # Data format conversion
    train_sourcecode_x = torch.tensor(train_sourcecode_x).float()
    train_bytecode_x = torch.tensor(train_bytecode_x).float()
    train_sourcecode_y = train_sourcecode_y.tolist()
    train_sourcecode_y = [int(num) for num in train_sourcecode_y]
    train_sourcecode_y = torch.tensor(train_sourcecode_y).float()

    # training data loader
    torch_dataset = Data.TensorDataset(train_sourcecode_x, train_bytecode_x, train_sourcecode_y)
    train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=args.batch_size, shuffle=args.shuffle,
                                   num_workers=2)

    #######################################  load test data  #######################################

    # load source code and bytecode of testing set respectively
    load_sourcecode = open(eval_sourcecode_path, 'r')
    eval_sourcecode_dict = json.load(load_sourcecode)
    # eval_sourcecode_names = np.array([item['contract_name'] for item in sourcecode_dict])
    eval_sourcecode_x = np.array([item['node_features'] for item in eval_sourcecode_dict])
    eval_sourcecode_y = np.array([item['targets'] for item in eval_sourcecode_dict])
    load_sourcecode.close()

    load_bytecode = open(eval_bytecode_path, 'r')
    eval_bytecode_dict = json.load(load_bytecode)
    # eval_bytecode_names = np.array([item['contract_name'] for item in bytecode_dict])
    eval_bytecode_x = np.array([item['node_features'] for item in eval_bytecode_dict])
    # eval_bytecode_y = np.array([item['targets'] for item in bytecode_dict])
    load_bytecode.close()

    # Data format conversion
    eval_sourcecode_x = torch.tensor(eval_sourcecode_x).float()
    eval_bytecode_x = torch.tensor(eval_bytecode_x).float()
    eval_sourcecode_y = eval_sourcecode_y.tolist()
    eval_sourcecode_y = [int(num) for num in eval_sourcecode_y]
    eval_sourcecode_y = torch.tensor(eval_sourcecode_y).float()

    # training data loader
    torch_dataset = Data.TensorDataset(eval_sourcecode_x, eval_bytecode_x, eval_sourcecode_y)
    test_loader = Data.DataLoader(dataset=torch_dataset, batch_size=args.batch_size, shuffle=args.shuffle,
                                  num_workers=2)
    
    input_size = 512 # Specify based on your dataset
    hidden_size = 128  # Adjust as needed
    output_size = 2 # Specify based on your dataset
    learning_rate = 0.001
    num_epochs = 50
    num_layer = 2
    num_head = 1
    a = 0.25
    b = 2
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = TestClassifier(sourcecode_input, bytecode_input, output_size).to(device)
    model = LSTMClassifier(input_size,hidden_size,num_layer,output_size,num_head).to(device)
    # model =  SimpleCNN(64,2).to(device)
    
    # models =  CustomModel(input_size,hidden_size,output_size).to(device)
    criterion = nn.CrossEntropyLoss().to(device) #适用与多分类任务
    # criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_S = []
    columns = ['Epoch']
    dfLoss = pd.DataFrame(columns=columns)
    for epoch in range(num_epochs):
        for (iteration, (train_batch_sourcecode_x, train_batch_bytecode_x, train_batch_y)) in enumerate(train_loader):
            # inputs, inputs2,targets = batch
            train_batch_bytecode_x = train_batch_bytecode_x.to(device)
            train_batch_sourcecode_x = train_batch_sourcecode_x.to(device)
            # print(len(train_batch_sourcecode_x))
            
            new = torch.cat((train_batch_sourcecode_x,train_batch_bytecode_x),dim=1).to(device)
            # print(new.shape)
            optimizer.zero_grad()
            # outputs = model(train_batch_sourcecode_x.to(device),train_batch_bytecode_x.to(device))
            outputs = model(new)
            # outputs = outputs.squeeze()
            # probabilities = F.softmax(outputs, dim=1) #打印出来预测概率
            train_batch_y = train_batch_y.to(outputs.device) 
            # loss_S = a * (1 - probabilities) * torch.log(probabilities) + (1 - a) * torch.pow(probabilities, b) * torch.log(1 - probabilities)
            
            # loss = criterion(outputs, train_batch_y.long()) - loss_S.sum()
            loss = criterion(outputs, train_batch_y.long())
          
           
         
            # print(probabilities.shape)
            
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
        loss_S.append(loss)

    prob_predictions = []
    true_label = []
    model.eval()
    
    with torch.no_grad():
        correct = 0
        total = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        all_predictions = []
        all_labels = []

    for (step, (eval_batch_sourcecode_x, eval_batch_bytecode_x, eval_batch_y)) in enumerate(test_loader):
        newtest = torch.cat((eval_batch_sourcecode_x,eval_batch_bytecode_x),dim=1).to(device)
        # outputs = model(eval_batch_sourcecode_x.to(device),eval_batch_bytecode_x.to(device))
        eval_batch_bytecode_x = eval_batch_bytecode_x.to(device)
        eval_batch_sourcecode_x = eval_batch_sourcecode_x.to(device)
        outputs = model(newtest)
        
        
        prob = torch.sigmoid(outputs).squeeze().detach().cpu().numpy()
        prob_predictions.extend(prob)
        true_label.extend(eval_batch_y.cpu().numpy())
        
        _, predicted = torch.max(outputs.data, 1)
        eval_batch_y = eval_batch_y.to(outputs.device)
        total += eval_batch_y.size(0)
        correct += (predicted == eval_batch_y).sum().item()
        
        tp += ((predicted == 1) & (eval_batch_y == 1)).sum().item()
        tn += ((predicted == 0) & (eval_batch_y == 0)).sum().item()
        fp += ((predicted == 1) & (eval_batch_y == 0)).sum().item()
        fn += ((predicted == 0) & (eval_batch_y == 1)).sum().item()
        
        all_predictions.extend(outputs[:, 1].detach().cpu().numpy())  # Assuming the positive class is at index 1
        all_labels.extend(eval_batch_y.cpu().numpy())
    print(tp,tn,fp,fn)
    print(len(all_labels))
    
    prob_predictions = np.array(prob_predictions)
    true_label = np.array(true_label) # caculate k-s
    # print(all_predictions)
    # Calculate AUC
    auc_score = roc_auc_score(all_labels, all_predictions)
    print(f'AUC Score: {auc_score}')
    
    # Calculate FPR and TPR for ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_predictions)
    
    roc_data = pd.DataFrame({
        'FPR':fpr,
        'TPR':tpr,
    })
    
    accuracy = correct / total
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    print(f'准确率：{accuracy * 100:.2f}%')
    print(f'精确度：{precision * 100:.2f}%')
    print(f'召回率：{recall * 100:.2f}%')
    print(f'F1 分数：{f1_score * 100:.2f}%')
    
    num_points = 500  
    bins = np.linspace(0, 1, num=num_points)
    
    # print(prob_predictions)
    
    positive_probs = prob_predictions[true_label==1]
    print(positive_probs)
    negative_probs = prob_predictions[true_label==0]
    print("----------------")
    print(negative_probs)
    print("----------------")
    
    
    positive_probs = positive_probs.flatten()
    negative_probs = negative_probs.flatten()
    
    counts_positive, bin_edges = np.histogram(positive_probs, bins=bins, density=True)
    counts_negative, _ = np.histogram(negative_probs, bins=bins, density=True)

    positive_cdf = np.cumsum(counts_positive) * np.diff(bin_edges)
    print(positive_cdf)
    print("----------------")
    negative_cdf = np.cumsum(counts_negative) * np.diff(bin_edges)
    print(negative_cdf)

    middle_points = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(middle_points, positive_cdf, label='Positive Class',color='blue')
    plt.plot(middle_points, negative_cdf, label='Negative Class',color='green')
    
    ks_statistic, p_value = ks_2samp(positive_probs, negative_probs)
    print(f'KS statistic: {ks_statistic}')
    
    max_diff = np.max(np.abs(positive_cdf - negative_cdf))
    index_of_max_diff = np.argmax(np.abs(positive_cdf - negative_cdf))

    plt.axvline(x=middle_points[index_of_max_diff], color='k', linestyle='--')
    plt.title(f'KS Statistic Plot (KS = {ks_statistic:.2f})')
    plt.xlabel('Score')

    plt.ylabel('Cumulative Probability')

    # 显示图例
    plt.legend()

    # 显示图表
    plt.show()
    plt.savefig('ks.png')
        
    
if __name__ == '__main__':
    main()