import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import tqdm
import nni
from LSTM import *
from dataset import FaceDataset

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


def prepare(args):
    global batch_size
    global learning_rate
    global num_layers
    global model
    global train_loader
    global test_loader
    global train_set
    global test_set
    train_set = FaceDataset(files_path=train_X)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_set = FaceDataset(files_path=test_X)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    learning_rate = args["lr"]
    if args["model"] == "lstm":
        model = LSTM(hidden_layer_size=args["hidden_layer_size"], num_layers=args["num_layers"]).to(device)
    elif args["model"] == "gru":
        model = GRU(hidden_layer_size=args["hidden_layer_size"], num_layers=args["num_layers"]).to(device)







def main():
    prepare(nni_config)
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
                print(total_loss)
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
                # torch.save(model.state_dict(), './gru.pth')
            nni.report_intermediate_result(acc)
            nni.report_final_result(best_acc)


if __name__ == "__main__":
    main()
