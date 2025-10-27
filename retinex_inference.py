import os
import cv2
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from glob import glob
from skimage import img_as_ubyte
from basicsr.models import create_model
from basicsr.utils.options import parse

# -----------------------------
# Self-ensemble (opzionale)
# -----------------------------
def self_ensemble(x, model):
    def forward_transformed(x, hflip, vflip, rotate, model):
        if hflip: x = torch.flip(x, (-2,))
        if vflip: x = torch.flip(x, (-1,))
        if rotate: x = torch.rot90(x, dims=(-2, -1))
        x = model(x)
        if rotate: x = torch.rot90(x, dims=(-2, -1), k=3)
        if vflip: x = torch.flip(x, (-1,))
        if hflip: x = torch.flip(x, (-2,))
        return x

    results = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                results.append(forward_transformed(x, hflip, vflip, rot, model))
    return torch.mean(torch.stack(results), dim=0)

# -----------------------------
# Inference su singola immagine
# -----------------------------
def enhance_image(img_path, model, device, output_path, use_self_ensemble=False):
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Immagine non trovata: {img_path}")
        return
    w, h = 400, 600
    img = cv2.resize(img, (h, w))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(img) / 255.0

    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    # padding per multipli di 4
    _, _, h, w = img_tensor.shape
    H, W = ((h + 4) // 4) * 4, ((w + 4) // 4) * 4
    padh, padw = H - h, W - w
    img_tensor = F.pad(img_tensor, (0, padw, 0, padh), mode='reflect')

    with torch.inference_mode():
        if use_self_ensemble:
            enhanced = self_ensemble(img_tensor, model)
        else:
            enhanced = model(img_tensor)

    enhanced = enhanced[:, :, :h, :w]
    enhanced = torch.clamp(enhanced, 0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()

    enhanced_bgr = cv2.cvtColor(img_as_ubyte(enhanced), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, enhanced_bgr)
    print(f"✅ Immagine migliorata salvata in: {output_path}")

# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Retinexformer - Low-light Image Enhancement")
    parser.add_argument("--opt", type=str, default="Options/RetinexFormer_SDSD_indoor.yml", help="File YAML opzioni")
    parser.add_argument("--weights", type=str, default="pretrained_weights/SDSD_indoor.pth", help="File pesi del modello")
    parser.add_argument("--input", type=str, required=True, help="Immagine singola o cartella di immagini")
    parser.add_argument("--output", type=str, default="./enhanced", help="Cartella output o immagine output")
    parser.add_argument("--self_ensemble", action="store_true", help="Usa self-ensemble per risultati migliori")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carica modello
    opt = parse(args.opt, is_train=False)
    opt['dist'] = False
    model = create_model(opt).net_g
    checkpoint = torch.load(args.weights, map_location=device)
    try:
        model.load_state_dict(checkpoint['params'])
    except:
        new_checkpoint = { 'module.'+k: v for k, v in checkpoint['params'].items() }
        model.load_state_dict(new_checkpoint)

    model.to(device).eval()
    print("✅ Modello caricato ed eseguito su", device)

    # Singola immagine
    if os.path.isfile(args.input):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        output_path = args.output if args.output.endswith(".png") or args.output.endswith(".jpg") else os.path.join(args.output, "enhanced.png")
        enhance_image(args.input, model, device, output_path, args.self_ensemble)

    # Cartella di immagini
    elif os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        image_paths = sorted(glob(os.path.join(args.input, "*.jpg")) + glob(os.path.join(args.input, "*.png")))
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            output_path = os.path.join(args.output, filename)
            enhance_image(img_path, model, device, output_path, args.self_ensemble)
    else:
        print("❌ Errore: percorso input non valido.")

if __name__ == "__main__":
    main()
