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
        # 如果路径不存在，就创建
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


# def load_npz_data(data_name):
#     with np.load(data_name,allow_pickle=True) as f:
#         train_x, train_y = [f[i] for i in f.files]
#     return train_x, train_y


def load_npz_data(data_name):
    loaded_data = np.load(data_name)

    loaded_split_data_list = []
    loaded_split_labels_list = []

    # Infer number of clients from loaded keys
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



# 绘制ROC曲线，计算AUC值
def ROC_AUC_Result(label_values, predict_values, reverse=False):
    if reverse:
        pos_label = 0  # 反转一下曲线（用于loss等指标，值越大，越猜为非成员）
        print('AUC = {}'.format(1 - roc_auc_score(label_values, predict_values)))
    else:
        pos_label = 1
        print('AUC = {}'.format(roc_auc_score(label_values, predict_values)))
    fpr, tpr, thresholds = roc_curve(label_values, predict_values,
                                     pos_label=pos_label)  # roc_auc_score函数返回曲线下面积，输入和roc_curve函数一样，第一个参数是真实label，第二个参数是预测值（处理后的指标值）
    # tpr也叫做recall
    print("Thresholds are {}. The len of Thresholds is {}".format(thresholds, len(thresholds)))
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic(ROC)')
    plt.plot(fpr, tpr, 'b', label='AUC=%0.4f' % roc_auc)  # 图示部分，显示了AUC的分值。
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.clf()
    # plt.show()

import numpy as np

def acc_with_quantile_threshold(scores, labels, n, reverse=True):
    """
    使用 n% 分位数作为阈值，计算分类准确率（ACC）
    
    Parameters:
    -----------
    scores : list 或 np.ndarray
        攻击得分，shape (n_samples,)
    labels : list 或 np.ndarray
        真实标签，1=成员, 0=非成员，shape (n_samples,)
    n : float
        分位数百分比，范围 [0, 100]
        例如：n=30 → 使用 30% 分位数作为阈值
    reverse : bool
        - True:  得分越大 → 越可能是成员（异常）
        - False: 得分越小 → 越可能是成员

    Returns:
    --------
    acc : float
        准确率
    threshold : float
        所使用的阈值
    pred_labels : np.ndarray
        预测标签
    """
    scores = np.array(scores)
    labels = np.array(labels)
    
    # 步骤1：计算 n% 分位数
    threshold = np.percentile(scores, n)
    
    # 步骤2：根据 reverse 确定“成员”的方向
    if reverse:
        # 得分 > threshold → 判为成员 (1)
        pred_labels = (scores > threshold).astype(int)
    else:
        # 得分 < threshold → 判为成员 (1)
        pred_labels = (scores < threshold).astype(int)
    
    # 步骤3：计算准确率
    acc = (pred_labels == labels).mean()
    print(f"acc is:{acc}")
    
    return acc, threshold, pred_labels

# def ROC_AUC_Result_logshow(label_values, predict_values, reverse=False):
#     if reverse:
#         pos_label = 0  # 反转一下曲线（用于loss等指标，值越大，越猜为非成员）
#         print('AUC = {}'.format(1 - roc_auc_score(label_values, predict_values)))
#     else:
#         pos_label = 1
#         print('AUC = {}'.format(roc_auc_score(label_values, predict_values)))
#     fpr, tpr, thresholds = roc_curve(label_values, predict_values,
#                                      pos_label=pos_label)  # roc_auc_score函数返回曲线下面积，输入和roc_curve函数一样，第一个参数是真实label，第二个参数是预测值（处理后的指标值）
#     # tpr也叫做recall
#     # print("Thresholds are {}. The len of Thresholds is {}".format(thresholds, len(thresholds)))
#     roc_auc = auc(fpr, tpr)
#     plt.title('Receiver Operating Characteristic(ROC)')
#     # plt.plot(fpr,tpr,'b', label='AUC=%0.4f' %roc_auc)   #图示部分，显示了AUC的分值。
#     plt.loglog(fpr, tpr, 'b', label='AUC=%0.4f' % roc_auc)
#     plt.legend(loc='lower right')
#     plt.plot([0.001, 1], [0.001, 1], 'r--')
#     plt.xlim([0.001, 1.0])
#     plt.ylim([0.001, 1.0])
#     plt.ylabel('TPR')
#     plt.xlabel('FPR')


#     # 读取曲线上的值
#     ax = plt.gca()
#     # find the line object that represents the ROC curve
#     line = ax.lines[0]
#     # get the x and y data of the line (FPR and TPR)
#     xdata = line.get_xdata()
#     ydata = line.get_ydata()
#     # create a interpolation function using xdata and ydata #ROC曲线的原始值是离散的，一定要进行插值。
#     f = interp1d(xdata, ydata)
#     # fpr_0 is your specified FPR value
#     fpr_0 = 0.001

#     tpr_0 = f(fpr_0)

#     print('TPR at 0.001 FPR is {}'.format(tpr_0))
#     # plt.show()

#     low = tpr[np.where(fpr<.001)[0][-1]]   #来自lira
#     print(f'TPR at 0.001 FPR is {low}  ----- lira')
#     plt.clf()
#     return tpr_0

    # plt.show()
def ROC_AUC_Result_logshow_with_auc(label_values, predict_values, reverse=False):
    if reverse:
        pos_label = 0  
        print('AUC = {}'.format(1 - roc_auc_score(label_values, predict_values)))
    else:
        pos_label = 1
        print('AUC = {}'.format(roc_auc_score(label_values, predict_values)))
        
    fpr, tpr, thresholds = roc_curve(label_values, predict_values, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    
    # 绘图部分
    plt.title('Receiver Operating Characteristic(ROC)')
    plt.loglog(fpr, tpr, 'b', label='AUC=%0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0.001, 1], [0.001, 1], 'r--')
    plt.xlim([0.001, 1.0])
    plt.ylim([0.001, 1.0])
    plt.ylabel('TPR')
    plt.xlabel('FPR')

    # 获取曲线数据进行插值
    ax = plt.gca()
    line = ax.lines[0]
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    
    # 使用线性插值处理离散点
    f = interp1d(xdata, ydata)

    # --- 新增内容：打印多个指定 FPR 处的 TPR ---
    target_fprs = [0.001, 0.005, 0.01, 0.05, 0.1]
    tpr_results = {}

    print("-" * 30)
    for f_val in target_fprs:
        # 确保请求的 FPR 在数据范围内，防止插值报错
        if f_val >= xdata.min() and f_val <= xdata.max():
            t_val = f(f_val)
            tpr_results[f_val] = t_val
            print(f'TPR at {f_val} FPR is {t_val:.6f}')
        else:
            print(f'FPR {f_val} is out of bounds for interpolation.')
    print("-" * 30)
    # ---------------------------------------

    # 保留你原来的 Lira 方式打印 (作为对比)
    low_001 = tpr[np.where(fpr < .001)[0][-1]]
    print(f'TPR at 0.001 FPR (Lira method) is {low_001}')

    plt.clf()
    # 返回 AUC 和 0.001 处的 TPR（如果需要返回多个，可以修改此处）
    return roc_auc, tpr_results.get(0.001)

def custom_collate(batch):
    """
    为了应对变长序列中的元素数量不一致的保存问题
    """
    # batch 是一个列表，每个元素是 (data, label, member)
    datas = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    members = [item[2] for item in batch]

    # 处理 datas 和 labels：转换为 Tensor，允许变长
    # 例如，如果 datas 是变长序列，直接返回列表
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
    
    # 绘图部分
    plt.title('Receiver Operating Characteristic(ROC)')
    plt.loglog(fpr, tpr, 'b', label='AUC=%0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0.001, 1], [0.001, 1], 'r--')
    plt.xlim([0.001, 1.0])
    plt.ylim([0.001, 1.0])
    plt.ylabel('TPR')
    plt.xlabel('FPR')

    # 获取曲线数据进行插值
    ax = plt.gca()
    line = ax.lines[0]
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    
    # 使用线性插值处理离散点
    f = interp1d(xdata, ydata)

    # --- 新增内容：打印多个指定 FPR 处的 TPR ---
    target_fprs = [0.001, 0.005,0.01, 0.05, 0.1]
    tpr_results = {}

    print("-" * 30)
    for f_val in target_fprs:
        # 确保请求的 FPR 在数据范围内，防止插值报错
        if f_val >= xdata.min() and f_val <= xdata.max():
            t_val = f(f_val)
            tpr_results[f_val] = t_val
            print(f'TPR at {f_val} FPR is {t_val:.6f}')
        else:
            print(f'FPR {f_val} is out of bounds for interpolation.')
    print("-" * 30)
    # ---------------------------------------

    # 保留你原来的 Lira 方式打印 (作为对比)
    low_001 = tpr[np.where(fpr < .001)[0][-1]]
    print(f'TPR at 0.001 FPR (Lira method) is {low_001}')

    plt.clf()
    # 返回 AUC 和 0.001 处的 TPR（如果需要返回多个，可以修改此处）
    return tpr_results.get(0.001)


def model_eval(self,model,dataloader,idx):
    """
    攻击模型的结果测试
    """
    model.eval()
    with torch.no_grad():
        total, correct = 0, 0
        for _, (x, y, _) in enumerate(dataloader):
            x = x.to(self.args.device)
            y = y.to(self.args.device)
            outputs = model(x)
            # max=torch.max(features)
            _, predicted = torch.max(outputs, 1)  # 获取最大值的索引（预测类别）
            total += y.size(0)  # 样本总数
            correct += (predicted == y).sum().item()
        print(f"----model:{idx}, Acc{correct / total}----")


def calculate_acc(target_scores, target_labels, cal_scores, cal_labels, 
                 method='best_acc', percentile=0.75,reverse=False):
    """
    根据额外的校准数据 (Calibration Data) 确定阈值，并计算目标数据的攻击准确率。
    
    原理：
    1. 使用 cal_scores 和 cal_labels (影子数据/辅助数据) 学习最佳阈值。
       这模拟了论文中攻击者通过查询影子模型或利用背景知识来估计误差分布的过程 [cite: 200, 683]。
    2. 将学到的阈值应用到 target_scores (受害者数据) 上进行评估。
    
    参数:
    - target_scores: list/array, 要攻击的目标样本得分 (Loss)
    - target_labels: list/array, 目标样本真实标签
    - cal_scores:    list/array, 用于确定阈值的辅助样本得分 (Loss)
    - cal_labels:    list/array, 辅助样本标签 (1=Member, 0=Non-Member)
    - method:        str, 
                     'best_acc'   -> 在校准集上寻找使 Acc 最高的阈值 (对应论文 Adversary 2 的最优策略)
                     'percentile' -> 在校准集上寻找非成员的特定百分位 (控制 FPR)
    - percentile:    float, 当 method='percentile' 时使用 (如 1 表示 1% FPR)
    
    返回:
    - result_dict: 包含在 target 数据集上的评估结果
    """
    if reverse:
        target_scores=-np.array(target_scores)
        cal_scores=-np.array(cal_scores)
    # 转换数据格式
    target_scores = np.array(target_scores)
    target_labels = np.array(target_labels)
    cal_scores = np.array(cal_scores)
    cal_labels = np.array(cal_labels)
    
    best_threshold = 0.0
    
    print(f"--- Threshold Selection Strategy: {method} ---")
    print(f"Calibration Data Size: {len(cal_scores)} (Members: {sum(cal_labels)}, Non-Members: {len(cal_labels)-sum(cal_labels)})")

    # --- 策略 1: 在校准集上暴力搜索最佳准确率的阈值 ---
    if method == 'best_acc':
        # 候选阈值来自于校准集的 Loss 值
        thresholds = np.unique(cal_scores)
        # 筛选非成员阈值
        n_scores = cal_scores[cal_labels == 0]
        # 2. 从筛选出的成员得分中提取唯一的阈值
        n_thresholds = np.unique(n_scores)
        best_acc_on_cal = -1
        
        # 遍历所有可能的阈值
        for t in thresholds:
            # 在校准集上预测
            preds = (cal_scores >= t).astype(int)
            acc = accuracy_score(cal_labels, preds)
            if acc > best_acc_on_cal:
                best_acc_on_cal = acc
                best_threshold = t
            percentile_rank = stats.percentileofscore(n_thresholds, best_threshold, kind='rank')
        
        print(f"Best Acc on Calibration Data: {best_acc_on_cal:4f}")
        print(f"percentile_rank: {percentile_rank}")

    # --- 策略 2: 在校准集上利用非成员分布确定阈值 ---
    elif method == 'percentile':
        # 提取校准集中的非成员 Loss
        cal_non_members = cal_scores[cal_labels == 0]
        
        if len(cal_non_members) == 0:
            raise ValueError("校准数据集中没有非成员 (Label=0)，无法使用 percentile 方法")
            
        # 计算校准集中非成员的百分位
        # 设定阈值使得只有 x% 的校准集非成员被误判为成员
        best_threshold = np.percentile(cal_non_members, percentile)
        
        print(f"Threshold set at {percentile}% percentile of Calibration Non-Members.")
    
    else:
        raise ValueError("Unknown method. Use 'best_acc' or 'percentile'.")

    print(f"Determined Threshold: {best_threshold:.6f}")

    # --- 最终评估 (应用到目标数据) ---
    # 使用在 Cal 数据上确定的阈值，攻击 Target 数据
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
    """
    计算在【最佳 F1 分数】对应阈值下的 Precision, Recall 和 Accuracy。
    
    参数:
    - y_true: 真实标签
    - y_score: 预测得分 (越高代表越像成员/正类)
               如果是 Loss，请传入 -Loss。
    
    返回:
    - best_f1: 最佳 F1 分数
    - p_at_best_f1: 该阈值下的 Precision
    - r_at_best_f1: 该阈值下的 Recall
    - acc_at_best_f1: 该阈值下的 Accuracy
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    # 1. 计算 ROC 曲线数据
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    
    # 2. 计算各项指标的列表 (基于平衡假设)
    # Precision = TPR / (TPR + FPR)
    with np.errstate(divide='ignore', invalid='ignore'):
        precision_list = tpr / (tpr + fpr)
    precision_list = np.nan_to_num(precision_list) 
    
    # Recall = TPR
    recall_list = tpr
    
    # Accuracy = (TPR + TNR) / 2 = (TPR + (1 - FPR)) / 2
    accuracy_list = (tpr + (1 - fpr)) / 2
    
    # F1 = 2 * P * R / (P + R)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_list = 2 * (precision_list * recall_list) / (precision_list + recall_list)
    f1_list = np.nan_to_num(f1_list)
    
    # --- 3. 找到 F1 最大的那个位置 (最佳操作点) ---
    best_idx = np.argmax(f1_list)
    
    # --- 4. 提取该位置下的所有指标 ---
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

def plot_roc(labels, scores, title='ROC Curve', label_name='Model', color='b',path='./reports',name='1.jpg'):
    """
    绘制并美化 ROC 曲线，自动计算并打印关键指标
    """
    path_and_name = os.path.join(path,name)
    # 1. 计算基础指标
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # 2. 准备插值函数 (用于获取特定 FPR 处的 TPR)
    # 增加一个小 epsilon 防止 log(0) 报错
    f_interp = interp1d(fpr, tpr, kind='linear', fill_value="extrapolate")
    
    # 定义关注的低误报率点
    target_fprs = [0.001, 0.01, 0.05, 0.1]
    print(f"--- {label_name} 关键指标 ---")
    print(f"Overall AUC: {roc_auc:.4f}")
    for f in target_fprs:
        t_val = f_interp(f)
        print(f"TPR at FPR={f:<5}: {t_val:.6f}")
    print("-" * 25)

    # 3. 开始绘图
    plt.figure(figsize=(8, 6))
    
    # 绘制主曲线
    plt.plot(fpr, tpr, color=color, lw=2, label=f'{label_name} (AUC = {roc_auc:.4f})')
    
    # 绘制对角基准线
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

    # 4. 样式美化
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # 如果需要查看低 FPR 细节，可以取消下面这一行的注释（切换到对数坐标）
    plt.xscale('log') 
    plt.yscale('log') 

    plt.tight_layout()
    # plt.savefig('roc_result.png') # 如需保存图片
    plt.savefig(path_and_name)

    return roc_auc


def plot_log_roc(labels, scores, title='Log-Log ROC Curve', label_name='Model', color='b', path='./reports', name='1.jpg'):
    """
    在双对数坐标系下绘制并美化 ROC 曲线
    """
    if not os.path.exists(path):
        os.makedirs(path)
    path_and_name = os.path.join(path, name)

    # 1. 计算基础指标
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # 2. 准备插值函数
    f_interp = interp1d(fpr, tpr, kind='linear', fill_value="extrapolate")
    
    # 定义关注的低误报率点
    target_fprs = [0.001, 0.01, 0.05, 0.1]
    print(f"--- {label_name} 关键指标 ---")
    print(f"Overall AUC: {roc_auc:.4f}")
    for f in target_fprs:
        t_val = f_interp(f)
        print(f"TPR at FPR={f:<5}: {t_val:.6f}")
    print("-" * 25)

    # 3. 开始绘图
    plt.figure(figsize=(8, 7)) # 对数图建议稍微方一点
    
    # 核心修改：使用 loglog 绘制主曲线
    plt.loglog(fpr, tpr, color=color, lw=2, label=f'{label_name} (AUC = {roc_auc:.4f})')
    
    # 绘制对角基准线 (在对数坐标下依然是 y=x)
    plt.loglog([1e-5, 1], [1e-5, 1], color='gray', lw=1, linestyle='--', label='Random Guess')

    # 4. 样式美化
    # 注意：对数坐标轴范围不能包含 0
    plt.xlim([1e-3, 1.0]) 
    plt.ylim([1e-3, 1.05])
    
    plt.xlabel('False Positive Rate (FPR) - Log Scale')
    plt.ylabel('True Positive Rate (TPR) - Log Scale')
    plt.title(title)
    plt.legend(loc="lower right")
    
    # 开启对数坐标轴的网格线 (包括主网格和次网格)
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.savefig(path_and_name)
    print(f"Result saved to: {path_and_name}")
    
    # 如果在 notebook 中运行可以取消注释查看，如果是脚本运行建议关闭
    # plt.show() 
    plt.close() # 记得关闭，防止连续调用时内存溢出

    return roc_auc


