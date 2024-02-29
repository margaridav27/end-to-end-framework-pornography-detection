import torch
import torch.nn.functional as funct

from captum.attr import (
  visualization as viz,
  IntegratedGradients,
  GuidedGradCam,
  LayerGradCam,
  LayerAttribution,
  LRP,
  LayerLRP
)

from src.utils.data import load_split, get_transforms
from src.utils.model import init_model
from src.datasets.pornography_frame_dataset import PornographyFrameDataset


MODEL_NAME = "resnet50"
INPUT_SHAPE = 224

data_loc = "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-20"

state_dicts_loc = "baseline/pornography-2k/models"
state_dict_loc = f"{state_dicts_loc}/model_resnet50_freeze_True_epochs_20_batch_32_optim_sgd_optimized_False.pth"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

state_dict = torch.load(state_dict_loc, map_location=device)
model = init_model(MODEL_NAME)
model = torch.nn.DataParallel(model)
model.load_state_dict(state_dict)

df_test = load_split(data_loc, ["test"])["test"]
data_transforms = get_transforms(INPUT_SHAPE)["test"]
dataset = PornographyFrameDataset(data_loc, df_test, data_transforms)

idx = 200
tensor_frame, label = dataset[idx]
input = tensor_frame.unsqueeze(0)

model = model.eval()
output = model(input)
output = funct.softmax(output, 1)
pred_score, pred_label = torch.topk(output, 1)
print(f"Ground-truth: {label} | Predicted: {pred_label.squeeze().item()} ({pred_score.squeeze().item()})")

last_conv_layer = model.module.layer4[-1].conv3
grad_cam = GuidedGradCam(model, last_conv_layer)
lgc_attr = grad_cam.attribute(input, target=pred_label)
lgc_attr_upsampled = LayerAttribution.interpolate(lgc_attr, input.shape[2:])

print(lgc_attr.shape)
print(lgc_attr_upsampled.shape)
print(input.shape)

_ = viz.visualize_image_attr(
  lgc_attr_upsampled[0].cpu().permute(1,2,0).detach().numpy(),
  input.squeeze().cpu().permute(1,2,0).detach().numpy(),
  method="blended_heat_map",
  sign="all",
  show_color_bar=True
)