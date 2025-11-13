import onnxruntime as ort
import numpy as np
import cv2
import torch
from PIL import Image

class InferHyDAONNX():
    def __init__(self, onnx_path):
        self.size = (640, 640)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.classes = ['N/A', 'person', "car", "rider", "bus", "truck", "bike", "motor", "traffic light", "traffic sign"]
        self.mask_threshold = 0.5
        self.det_threshold = 0.8
        self.onnx_path = onnx_path
        
    def rescale_boxes(self, boxes, orig_size):
        """
        boxes: (N,4) in cxcywh normalized (0..1)
        Return xyxy in original resolution
        """
        orig_h, orig_w = orig_size
        cx, cy, w, h = boxes.T

        x1 = (cx - w/2) * orig_w
        y1 = (cy - h/2) * orig_h
        x2 = (cx + w/2) * orig_w
        y2 = (cy + h/2) * orig_h

        b = np.stack([x1, y1, x2, y2], axis=1)
        return b
    
    def postprocess_mask(self, mask, orig_size):
        """
        mask: (1,1,Hm,Wm) logits
        return: (H_orig, W_orig) uint8 binary mask
        """
        mask = torch.from_numpy(mask)
        mask = mask.sigmoid()

        # Resize to original size
        mask_up = torch.nn.functional.interpolate(
            mask,
            size=orig_size,
            mode="bilinear",
            align_corners=False
        ).squeeze()

        mask_bin = (mask_up > self.mask_threshold).cpu().numpy().astype(np.uint8)
        return mask_bin
        
    def preprocess_frame(self, frame):

        h0, w0 = frame.shape[:2]
        orig_size = (h0, w0)

        # Resize to model input
        frame_resized = cv2.resize(frame, self.size[::-1])  # (W,H)

        # Normalize
        frame_norm = frame_resized.astype(np.float32) / 255.0
        frame_norm = (frame_norm - self.mean) / self.std

        # To NCHW
        frame_tensor = frame_norm.transpose(2, 0, 1)[None, :]  # (1,3,H,W)
        return frame_tensor.astype(np.float32), orig_size, frame_resized

    def visualize(self, img_path, boxes, labels, scores, mask):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw boxes
        for (x1, y1, x2, y2), sc, lb in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f"{self.classes[lb]}:{sc:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Overlay mask
        mask_colored = np.zeros_like(img)
        mask_colored[:, :, 1] = mask * 255  # green

        overlay = cv2.addWeighted(img, 0.7, mask_colored, 0.3, 0)

        cv2.imshow("Result", overlay)
        cv2.waitKey(0)
    
    def run(self, frame_path):
        
        # Create ONNXRuntime session
        sess = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        # Input name
        input_name = sess.get_inputs()[0].name
        
        # Preprocess
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor, orig_size, frame_resized = self.preprocess_frame(frame)
        
        # Run
        pred_logits, pred_boxes, pred_masks = sess.run(
            None,
            {input_name: frame_tensor}
        )
        # Shape info
        print("pred_logits:", pred_logits.shape)  # (1,100,num_classes)
        print("pred_boxes :", pred_boxes.shape)   # (1,100,4)
        print("pred_masks :", pred_masks.shape)   # (1,1,Hm,Wm)
        
        scores = torch.from_numpy(pred_logits[0]).softmax(-1)[:, :-1]  # remove background
        max_scores, labels = scores.max(-1)

        keep = max_scores > self.det_threshold
        boxes = pred_boxes[0][keep]
        labels = labels[keep]
        scores = max_scores[keep]
        
        boxes_xyxy = self.rescale_boxes(boxes, orig_size)
        final_mask = self.postprocess_mask(pred_masks, orig_size)
        
        self.visualize(
            frame_path,
            boxes_xyxy, labels, scores, final_mask
        )

if __name__ == "__main__":
    onnx_path = "hyda_r50_e29.onnx"
    img_path = "data/bdd100k/bdd100k_images_100k/val/ca6412a2-3db85e24.jpg"

    infer = InferHyDAONNX(onnx_path)
    infer.run(img_path)