# @Author : LiZhongzheng
# 开发时间  ：2025-03-10 8:57

from train import *

"""测试数据集，并且保存预测结果"""


def test_process(dataloader, model, device):
    model.eval()
    preds = []
    for x in tqdm(dataloader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds


def save_pred(preds, file):
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_train = np.load('x_train.npy')  # 加载x_train
    model = MyModel(input=x_train.shape[1]).to(device)
    config = np.load('config.npy', allow_pickle=True).item()
    model.load_state_dict(torch.load(config['save_model']))
    test_loader = np.load('test_loader.npy', allow_pickle=True).item()
    preds = predict(test_loader, model, device)
    save_pred(preds, 'pred.csv')
