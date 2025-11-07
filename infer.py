import argparse
import math

from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
# import ipywidgets as widgets
from IPython.display import display, clear_output
from models import build_model
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False)

# COCO classes
CLASSES = [
    'N/A', 'person', "car", "rider", "bus", "truck", "bike", "motor", "traffic light", "traffic sign"
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize([640,640]),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax() - 1
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

# COLORS, CLASSES, v.v. giữ nguyên như bạn đang dùng

def plot_results_with_heatmap(
    pil_img,
    prob,                 # Tensor [N, num_classes] (đã softmax hoặc logits đều OK)
    boxes,                # Tensor [N, 4] ở pixel xyxy
    heatmap=None,         # Tensor logits hoặc prob của mask: [Hh,Wh] hoặc [1,1,Hh,Wh]
    thr=0.5,              # ngưỡng binary cho mask
    overlay_alpha=0.4,    # độ trong suốt heatmap/mask
    cmap='jet',
    show_binary=False     # nếu True: overlay thêm binary mask (thr)
):
    """
    - Vẽ bounding boxes + nhãn (giống plot_results cũ).
    - Nếu có heatmap: overlay heatmap lên ảnh.
      heatmap có thể là logits (chưa sigmoid) hoặc prob (0..1), hàm tự nhận dạng.
    - Option show_binary=True để overlay binary mask (thr).
    """
    # Chuẩn bị ảnh và axes
    fig = plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()

    # 1) Vẽ heatmap nếu có
    if heatmap is not None:
        hm = heatmap
        if isinstance(hm, torch.Tensor):
            hm = hm.detach().cpu()
            # chấp nhận [H,W] hoặc [1,1,H,W]
            if hm.dim() == 4:
                hm = hm.squeeze(0).squeeze(0)  # -> [H,W]
            # nếu là logits: đưa về prob
            if hm.max() > 1.0 or hm.min() < 0.0:
                hm = hm.sigmoid()
            hm_np = hm.numpy()  # [H,W] 0..1
        else:
            hm_np = hm  # numpy 2D

        # resize heatmap về kích thước ảnh PIL
        H, W = pil_img.size[1], pil_img.size[0]  # PIL: (W,H)
        hm_t = torch.from_numpy(hm_np).float().unsqueeze(0).unsqueeze(0)  # [1,1,Hh,Wh]
        hm_up = F.interpolate(hm_t, size=(H, W), mode='bilinear', align_corners=False).squeeze().numpy()

        # overlay heatmap
        plt.imshow(hm_up, cmap=cmap, alpha=overlay_alpha, vmin=0.0, vmax=1.0)

        # overlay binary mask (tuỳ chọn)
        if show_binary:
            mask_bin = (hm_up > thr).astype(np.float32)
            # dùng một lớp overlay xanh lá mờ
            green = np.zeros((H, W, 3), dtype=np.float32)
            green[..., 1] = 1.0
            plt.imshow(green, alpha=overlay_alpha * mask_bin, interpolation='nearest')

    # 2) Vẽ boxes + nhãn
    colors = COLORS * 100
    prob_soft = prob
    if isinstance(prob_soft, torch.Tensor):
        # nếu là logits: softmax
        if prob_soft.max() > 1.0 or prob_soft.min() < 0.0:
            prob_soft = prob_soft.softmax(-1)
        prob_soft = prob_soft.detach().cpu()

    for p, (xmin, ymin, xmax, ymax), c in zip(prob_soft, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        # background là lớp cuối → bỏ lớp bg
        p_no_bg = p[:-1] if p.numel() == len(CLASSES) + 1 else p
        cl = int(p_no_bg.argmax())
        score = float(p_no_bg[cl])
        name = CLASSES[cl] if 0 <= cl < len(CLASSES) else f'cls{cl}'
        text = f'{name}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    plt.tight_layout()
    plt.show()

    
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

def main(args):
    
    model, _ = build_model(args)
    state_dict = torch.load("checkpoint0099.pth", map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict['model'])
    model.eval()
    #data/bdd100k/bdd100k_images_100k/val/b1c9c847-3bda4659.jpg
    #data/coco/val2017/000000000872.jpg
    img_path = 'data/BDD/bdd100k/bdd100k_images_100k/val/b1c9c847-3bda4659.jpg'
    im = Image.open(img_path).convert('RGB')
    
    img = transform(im).unsqueeze(0)
    
    # img = img.repeat(2, 1, 1, 1)
    
    print(img.size())
    
    outputs = model(img)       
    
    print(outputs.keys())
    for item in outputs.keys():
        if(item != 'aux_outputs'):
            print(f"{item} size: {outputs[item].size()}")
            
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7
    
    print(outputs['pred_masks'])
    logits = outputs["pred_masks"][0, 0]         # [160,160], logits
    heat = logits.sigmoid().cpu()                # [160,160], 0..1
    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    
    scores = probas[keep]
    # plot_results(im, scores, bboxes_scaled)
    plot_results_with_heatmap(
        pil_img=im,
        prob=scores,         # tensor [N,C]
        boxes=bboxes_scaled,   # tensor [N,4]
        heatmap=outputs["pred_masks"],  # [Hh,Wh] hoặc [1,1,Hh,Wh]
        thr=0.45,
        overlay_alpha=0.45,
        cmap='jet',
        show_binary=False
    )
    

class Infer():
    def __init__(self, check_point):
        super().__init__()
        
        self.classes = ['N/A', 'person', "car", "rider", "bus", "truck", "bike", "motor", "traffic light", "traffic sign"]
        self.state_dict = torch.load(check_point, map_location='cpu', weights_only=False)
    
    def _load_model():
        pass
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR inference', parents=[get_args_parser()])
    args = parser.parse_args()
    args.masks = False
    main(args)