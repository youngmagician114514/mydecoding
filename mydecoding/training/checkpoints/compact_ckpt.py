# compact_ckpt.py
import torch

big_path = "dual_decoder_stageB_fusion_head2.pt"
small_path = "dual_decoder_stageB_student.pt"

print("Loading big checkpoint:", big_path)
state = torch.load(big_path, map_location="cpu")  # 这一步需要很多 RAM

# state 是一个 state_dict: {param_name: tensor}
student_state = {}
for name, tensor in state.items():
    if name.startswith("head1.") or name.startswith("fusion.") or name.startswith("head2."):
        student_state[name] = tensor

print("Keeping", len(student_state), "tensors for student only")
torch.save(student_state, small_path)
print("Saved compact checkpoint to:", small_path)
