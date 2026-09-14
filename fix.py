import pickle
import os
def fix(path):
    member_file = os.path.join(path,"member_losses.pkl")
    nonmember_file =os.path.join(path, "nonmember_losses.pkl")

    output_file = os.path.join(path,"metrics_history_selected.pkl")

    with open(member_file, "rb") as f:
        member_data = pickle.load(f)

    with open(nonmember_file, "rb") as f:
        nonmember_data = pickle.load(f)

    if isinstance(member_data, list):
        member_data = {"loss": member_data}
    if isinstance(nonmember_data, list):
        nonmember_data = {"loss": nonmember_data}

    full_metrics_history = {
        "member": member_data,
        "non_member": nonmember_data
    }

    with open(output_file, "wb") as f:
        pickle.dump(full_metrics_history, f)

    print(f"✅ 修复完成，已保存到 {output_file}")