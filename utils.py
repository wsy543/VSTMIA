import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve,auc
from scipy.interpolate import interp1d
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
from scipy import stats
from sklearn import metrics

def path_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")




def load_npz_data(data_name):
    loaded_data = np.load(data_name)

    loaded_split_data_list = []
    loaded_split_labels_list = []

    client_indices = set()
    for key in loaded_data.files:
        if key.startswith('client_') and key.endswith('_data'):
            try:
                idx_str = key[len('client_'):-len('_data')]
                client_idx = int(idx_str)
                client_indices.add(client_idx)
            except ValueError:
                pass

    num_clients = len(client_indices)
    sorted_client_indices = sorted(list(client_indices))

    if num_clients == 0:
        loaded_data.close()
        return [], []

    for client_idx in sorted_client_indices:
         data_key = f'client_{client_idx}_data'
         labels_key = f'client_{client_idx}_labels'

         loaded_split_data_list.append(loaded_data[data_key])
         loaded_split_labels_list.append(loaded_data[labels_key])

    loaded_data.close()

    return loaded_split_data_list, loaded_split_labels_list



import numpy as np

def acc_with_quantile_threshold(scores, labels, n, reverse=True):
    scores = np.array(scores)
    labels = np.array(labels)
    
    threshold = np.percentile(scores, n)
    
    if reverse:
        pred_labels = (scores > threshold).astype(int)
    else:
        pred_labels = (scores < threshold).astype(int)
    
    acc = (pred_labels == labels).mean()
    print(f"acc is:{acc}")
    
    return acc, threshold, pred_labels







def ROC_AUC_Result_logshow_with_auc(label_values, predict_values, reverse=False):
    if reverse:
        pos_label = 0  
        print('AUC = {}'.format(1 - roc_auc_score(label_values, predict_values)))
    else:
        pos_label = 1
        print('AUC = {}'.format(roc_auc_score(label_values, predict_values)))
        
    fpr, tpr, thresholds = roc_curve(label_values, predict_values, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    
    plt.title('Receiver Operating Characteristic(ROC)')
    plt.loglog(fpr, tpr, 'b', label='AUC=%0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0.001, 1], [0.001, 1], 'r--')
    plt.xlim([0.001, 1.0])
    plt.ylim([0.001, 1.0])
    plt.ylabel('TPR')
    plt.xlabel('FPR')

    ax = plt.gca()
    line = ax.lines[0]
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    
    f = interp1d(xdata, ydata)

    target_fprs = [0.001, 0.005, 0.01, 0.05, 0.1]
    tpr_results = {}

    print("-" * 30)
    for f_val in target_fprs:
        if f_val >= xdata.min() and f_val <= xdata.max():
            t_val = f(f_val)
            tpr_results[f_val] = t_val
            print(f'TPR at {f_val} FPR is {t_val:.6f}')
        else:
            print(f'FPR {f_val} is out of bounds for interpolation.')
    print("-" * 30)

    low_001 = tpr[np.where(fpr < .001)[0][-1]]
    print(f'TPR at 0.001 FPR (Lira method) is {low_001}')

    plt.clf()
    return roc_auc, tpr_results.get(0.001)

def custom_collate(batch):
    datas = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    members = [item[2] for item in batch]

    return {
        "data": [torch.from_numpy(x) for x in datas],
        "label": torch.tensor(labels),
        "member": torch.tensor(members)
    }


def ROC_AUC_Result_logshow(label_values, predict_values, reverse=False):
    if reverse:
        pos_label = 0  
        print('AUC = {}'.format(1 - roc_auc_score(label_values, predict_values)))
    else:
        pos_label = 1
        print('AUC = {}'.format(roc_auc_score(label_values, predict_values)))
        
    fpr, tpr, thresholds = roc_curve(label_values, predict_values, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    
    plt.title('Receiver Operating Characteristic(ROC)')
    plt.loglog(fpr, tpr, 'b', label='AUC=%0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0.001, 1], [0.001, 1], 'r--')
    plt.xlim([0.001, 1.0])
    plt.ylim([0.001, 1.0])
    plt.ylabel('TPR')
    plt.xlabel('FPR')

    ax = plt.gca()
    line = ax.lines[0]
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    
    f = interp1d(xdata, ydata)

    target_fprs = [0.001, 0.005,0.01, 0.05, 0.1]
    tpr_results = {}

    print("-" * 30)
    for f_val in target_fprs:
        if f_val >= xdata.min() and f_val <= xdata.max():
            t_val = f(f_val)
            tpr_results[f_val] = t_val
            print(f'TPR at {f_val} FPR is {t_val:.6f}')
        else:
            print(f'FPR {f_val} is out of bounds for interpolation.')
    print("-" * 30)

    low_001 = tpr[np.where(fpr < .001)[0][-1]]
    print(f'TPR at 0.001 FPR (Lira method) is {low_001}')

    plt.clf()
    return tpr_results.get(0.001)


def model_eval(self,model,dataloader,idx):
    model.eval()
    with torch.no_grad():
        total, correct = 0, 0
        for _, (x, y, _) in enumerate(dataloader):
            x = x.to(self.args.device)
            y = y.to(self.args.device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        print(f"----model:{idx}, Acc{correct / total}----")


def calculate_acc(target_scores, target_labels, cal_scores, cal_labels, 
                 method='best_acc', percentile=0.75,reverse=False):
    if reverse:
        target_scores=-np.array(target_scores)
        cal_scores=-np.array(cal_scores)
    target_scores = np.array(target_scores)
    target_labels = np.array(target_labels)
    cal_scores = np.array(cal_scores)
    cal_labels = np.array(cal_labels)
    
    best_threshold = 0.0
    
    print(f"--- Threshold Selection Strategy: {method} ---")
    print(f"Calibration Data Size: {len(cal_scores)} (Members: {sum(cal_labels)}, Non-Members: {len(cal_labels)-sum(cal_labels)})")

    if method == 'best_acc':
        thresholds = np.unique(cal_scores)
        n_scores = cal_scores[cal_labels == 0]
        n_thresholds = np.unique(n_scores)
        best_acc_on_cal = -1
        
        for t in thresholds:
            preds = (cal_scores >= t).astype(int)
            acc = accuracy_score(cal_labels, preds)
            if acc > best_acc_on_cal:
                best_acc_on_cal = acc
                best_threshold = t
            percentile_rank = stats.percentileofscore(n_thresholds, best_threshold, kind='rank')
        
        print(f"Best Acc on Calibration Data: {best_acc_on_cal:4f}")
        print(f"percentile_rank: {percentile_rank}")

    elif method == 'percentile':
        cal_non_members = cal_scores[cal_labels == 0]
        
        if len(cal_non_members) == 0:
            raise ValueError("校准数据集中没有非成员 (Label=0)，无法使用 percentile 方法")
            
        best_threshold = np.percentile(cal_non_members, percentile)
        
        print(f"Threshold set at {percentile}% percentile of Calibration Non-Members.")
    
    else:
        raise ValueError("Unknown method. Use 'best_acc' or 'percentile'.")

    print(f"Determined Threshold: {best_threshold:.6f}")

    final_preds = (target_scores >= best_threshold).astype(int)
    
    acc = accuracy_score(target_labels, final_preds)
    prec = precision_score(target_labels, final_preds, zero_division=0)
    rec = recall_score(target_labels, final_preds, zero_division=0)
    print(f"-------method: {method} Final Target Acc: {acc:.4f}-------")
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "threshold": best_threshold,
        "method": method
    }


def get_best_metrics(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        precision_list = tpr / (tpr + fpr)
    precision_list = np.nan_to_num(precision_list) 
    
    recall_list = tpr
    
    accuracy_list = (tpr + (1 - fpr)) / 2
    
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_list = 2 * (precision_list * recall_list) / (precision_list + recall_list)
    f1_list = np.nan_to_num(f1_list)
    
    best_idx = np.argmax(f1_list)
    
    best_f1 = f1_list[best_idx]
    p_at_best_f1 = precision_list[best_idx]
    r_at_best_f1 = recall_list[best_idx]
    acc_at_best_f1 = accuracy_list[best_idx]
    best_threshold = thresholds[best_idx]
    
    print(f"--- Metrics at Best F1 Threshold ---")
    print(f"Threshold     : {best_threshold:.6f}")
    print(f"Best F1       : {best_f1:.4f}")
    print(f"Accuracy      : {acc_at_best_f1:.4f}")
    print(f"Precision     : {p_at_best_f1:.4f}")
    print(f"Recall        : {r_at_best_f1:.4f}")
    
    return best_f1, p_at_best_f1, r_at_best_f1, acc_at_best_f1
