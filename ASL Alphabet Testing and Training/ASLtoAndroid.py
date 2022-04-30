import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
from torchvision.models import mobilenet_v2
model = mobilenet_v2(pretrained=True)
model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=37)
print(model.classifier)

path = "MobileASL.pth"
model.load_state_dict(torch.load(path))
model.eval()
example = torch.rand(1, 3, 50, 50)
traced_script_module = torch.jit.trace(model, example)
traced_script_module_optimized = optimize_for_mobile(traced_script_module)
traced_script_module_optimized._save_for_lite_interpreter("model.ptl")