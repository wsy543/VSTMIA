import pickle
import os
def fix(path):
    # 输入文件
    member_file = os.path.join(path,"member_losses.pkl")
    nonmember_file =os.path.join(path, "nonmember_losses.pkl")

    # 输出文件
    output_file = os.path.join(path,"metrics_history_selected.pkl")

    # 读取
    with open(member_file, "rb") as f:
        member_data = pickle.load(f)

    with open(nonmember_file, "rb") as f:
        nonmember_data = pickle.load(f)

    # 如果直接是 list，包成 {'loss': list}
    if isinstance(member_data, list):
        member_data = {"loss": member_data}
    if isinstance(nonmember_data, list):
        nonmember_data = {"loss": nonmember_data}

    # 合并成最终结构
    full_metrics_history = {
        "member": member_data,
        "non_member": nonmember_data
    }

    # 保存
    with open(output_file, "wb") as f:
        pickle.dump(full_metrics_history, f)

    print(f"✅ 修复完成，已保存到 {output_file}")