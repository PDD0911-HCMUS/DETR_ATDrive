import torch, onnx, onnxruntime as ort
from pathlib import Path

# 1) Khởi tạo & nạp trọng số
from models import build_model, detr  # <- đổi đúng path module của bạn

device = torch.device("cpu")  # hoặc "cuda"
model = detr().to(device).eval()

ckpt = torch.load("model.pth", map_location=device)
state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
model.load_state_dict(state, strict=True)

# 2) Dummy input: N=1, C=3, H=W=640
dummy = torch.randn(1, 3, 640, 640, device=device)

# 3) Wrapper chỉ trả về 3 output cần thiết
class ExportWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x):
        out = self.net(x)  # dict with keys: pred_logits, pred_boxes, pred_masks, aux_outputs
        # đảm bảo thứ tự trả ra đúng với output_names
        return out["pred_logits"], out["pred_boxes"], out["pred_masks"]

wrapped = ExportWrapper(model).to(device).eval()

onnx_path = "DetATDrive_infer.onnx"

# 4) Export
torch.onnx.export(
    wrapped,
    dummy,
    onnx_path,
    input_names=["images"],
    output_names=["pred_logits", "pred_boxes", "pred_masks"],
    opset_version=13,
    do_constant_folding=True,
    # Dynamic axes: batch + H,W để linh hoạt
    dynamic_axes={
        "images":       {0: "batch", 2: "in_h", 3: "in_w"},
        "pred_logits":  {0: "batch"},                 # [B, Q, C]
        "pred_boxes":   {0: "batch"},                 # [B, Q, 4] (cx,cy,w,h) dạng chuẩn của DETR
        "pred_masks":   {0: "batch", 2: "mask_h", 3: "mask_w"}  # [B, Q, Hm, Wm]
    }
)

print(f"✅ Exported ONNX: {onnx_path}")
