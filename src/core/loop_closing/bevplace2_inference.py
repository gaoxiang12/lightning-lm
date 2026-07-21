#!/usr/bin/env python3
"""
BEVPlace2 inference server for lightning-lm.
Reads BEV images from stdin, runs REIN model, returns descriptors.
Protocol: one JSON per line on stdin/stdout.
"""

import sys
import json
import struct
import numpy as np
import cv2

import torch
import torch.nn.functional as F

# Add BEVPlace2 to path
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'BEVPlace2'))

from REIN import REIN


class BEVPlace2Inference:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = REIN().to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.global_feat_dim = 8192  # 128 * 64

    def extract_descriptor(self, bev_image):
        """
        Input: bev_image - HxW uint8 grayscale
        Output: global descriptor (8192-dim L2-normalized)
        """
        img = bev_image.astype(np.float32) / 256.0
        img = img[np.newaxis, :, :].repeat(3, axis=0)  # [3, H, W]
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)  # [1, 3, H, W]

        with torch.no_grad():
            out1, local_feats, global_desc = self.model(tensor)

        return global_desc.cpu().numpy().flatten()

    def extract_descriptors_batch(self, bev_images):
        """
        Input: list of HxW uint8 grayscale images
        Output: Nx8192 descriptor matrix
        """
        descs = []
        for img in bev_images:
            d = self.extract_descriptor(img)
            descs.append(d)
        return np.array(descs, dtype=np.float32)


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'BEVPlace2',
                      'runs', 'Aug08_10-17-29', 'model_best.pth.tar')

    server = BEVPlace2Inference(model_path)

    # Signal ready
    print(json.dumps({"status": "ready", "dim": server.global_feat_dim}))
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"error": "invalid json"}))
            sys.stdout.flush()
            continue

        cmd = req.get("cmd", "")

        if cmd == "extract":
            # Extract descriptor for a single BEV image (sent as base64 or raw bytes)
            # For simplicity, expect raw bytes: height, width, then H*W uint8 values
            data_b64 = req.get("data", "")
            import base64
            raw = base64.b64decode(data_b64)
            h, w = struct.unpack('HH', raw[:4])
            img = np.frombuffer(raw[4:], dtype=np.uint8).reshape(h, w)

            desc = server.extract_descriptor(img)

            import base64
            desc_b64 = base64.b64encode(desc.tobytes()).decode('ascii')
            print(json.dumps({"desc": desc_b64, "dim": len(desc)}))

        elif cmd == "extract_batch":
            # Extract descriptors for multiple BEV images
            data_b64 = req.get("data", "")
            import base64
            raw = base64.b64decode(data_b64)
            n_images = struct.unpack('I', raw[:4])[0]
            offset = 4
            descs = []
            for _ in range(n_images):
                h, w = struct.unpack('HH', raw[offset:offset+4])
                offset += 4
                img = np.frombuffer(raw[offset:offset+h*w], dtype=np.uint8).reshape(h, w)
                offset += h * w
                desc = server.extract_descriptor(img)
                descs.append(desc)

            descs = np.array(descs, dtype=np.float32)
            desc_b64 = base64.b64encode(descs.tobytes()).decode('ascii')
            print(json.dumps({"descs": desc_b64, "n": n_images, "dim": len(descs[0])}))

        elif cmd == "ping":
            print(json.dumps({"pong": True}))

        else:
            print(json.dumps({"error": f"unknown command: {cmd}"}))

        sys.stdout.flush()


if __name__ == "__main__":
    main()
