import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import tqdm
import nni
from LSTM import *
from dataset import FaceDataset
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import numpy as np
nni_config = nni.get_next_parameter()
device = "cuda" if torch.cuda.is_available() else "cpu"

epoches = 200
batch_size = 16
learning_rate = 0.01
hidden_size = 64
num_layers = 1

f = open("final_data_list.txt", "r")
lines = f.readlines()
f.close()
lines = [line.strip().replace("\\", "/") for line in lines]
train_X, test_X, _, _ = train_test_split(lines, range(len(lines)), test_size=0.3, random_state=17)

train_set = FaceDataset(files_path=train_X)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_set = FaceDataset(files_path=test_X)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
learning_rate = 0.001
model = GRU().to(device)

    # learning_rate = args["lr"]
    # if args["model"] == "lstm":
    #     model = LSTM(hidden_layer_size=args["hidden_layer_size"], num_layers=args["num_layers"]).to(device)
    # elif args["model"] == "gru":
    #     model = GRU(hidden_layer_size=args["hidden_layer_size"], num_layers=args["num_layers"]).to(device)







def main():
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model_loss = nn.CrossEntropyLoss().to(device)
    best_acc = 0
    for epoch in range(epoches):
        with tqdm.tqdm(total=int(len(train_set) / batch_size), desc="Training") as t:
            total_loss = 0
            total_iter = 0
            total_acc = 0
            for i, (feature, label) in enumerate(train_loader):
                feature, label = feature.to(device), label.to(device)
                optimizer.zero_grad()
                outputs = model(feature)

                loss = model_loss(outputs, label)
                total_loss += loss.cpu().detach().numpy()
                _, predicted = torch.max(outputs.data, 1)
                total_acc += (predicted == label).sum()
                loss.backward()
                optimizer.step()
                total_iter = total_iter + batch_size
                t.set_postfix(
                    loss=total_loss / total_iter,
                    acc=total_acc.cpu().numpy() / total_iter,
                )
                t.update(1)
        with torch.no_grad():
            acc = 0
            total = 0
            for (feature_test, label_test) in test_loader:
                feature_test, label_test = feature_test.to(device), label_test.to(device)
                outputs = model(feature_test)
                _, predicted = torch.max(outputs.data, 1)
                total += label_test.size(0)
                acc += (predicted == label_test).sum()
            acc = 100 * acc / total
            print(acc.item(), "%")
            if best_acc < acc:
                best_acc = acc
                torch.save(model.state_dict(), './gru.pth')
            #nni.report_intermediate_result(acc)
            #nni.report_final_result(best_acc)


def predict(model, path):
    model.load_state_dict(torch.load(path))
    y_pred = None
    y_true = None
    for (feature_test, label_test) in test_loader:
        feature_test, label_test = feature_test.to(device), label_test.to(device)
        outputs = model(feature_test)
        layer = nn.Softmax(1)
        outputs = layer(outputs)
        y_pred = (
            outputs.data
            if y_pred == None
            else torch.cat((y_pred, outputs.data))
        )
        y_true = (
            label_test.data
            if y_true == None
            else torch.cat((y_true, label_test.data))
        )
    return y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()

def rocCurve(model, path):
    y_label, y_scores = predict(model, path)
    y_test = label_binarize(y_label, classes=[0, 1, 2])
    plt.figure()
    plt.title("ROC CURVE")
    for i in range(3):
        # 计算每个类别的FPR, TPR
        fpr, tpr, thr = roc_curve(y_test[:, i], y_scores[:, i])
        colors = ["r", "g", "b"]
        markers = ["--", "--", "--"]
        plt.plot(fpr, tpr, color=colors[i], linestyle=markers[i], label="level_{},AUC: {:.2f}".format(i, auc(fpr, tpr)))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
    fpr, tpr, thresholds = roc_curve(y_test.ravel(), y_scores.ravel())
    print(roc_auc_score(y_test, y_scores, average='micro'))
    plt.plot(fpr, tpr, linestyle="--", label="average,AUC: {:.2f}".format(roc_auc_score(y_test, y_scores, average='micro')))
    plt.legend(loc="lower right")
    cm = confusion_matrix(y_label, np.argmax(y_scores, 1))
    plt.figure()
    print(cm)
    plt.matshow(cm)
    print(classification_report(y_label, np.argmax(y_scores, 1)))
    print("AUC:", roc_auc_score(y_test, y_scores, multi_class="ovr", average=None))




if __name__ == "__main__":
    rocCurve(LSTM().to(device), './lstm.pth')
