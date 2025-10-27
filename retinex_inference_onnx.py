import os
import cv2
import argparse
import numpy as np
import time
from glob import glob
from skimage import img_as_ubyte
import onnxruntime as ort

# -----------------------------
# Self-ensemble (opzionale)
# -----------------------------
def self_ensemble_numpy(x, session):
    def forward_transformed(x, hflip, vflip, rotate, session):
        if hflip: x = np.flip(x, axis=3)  # flip width
        if vflip: x = np.flip(x, axis=2)  # flip height
        if rotate: x = np.rot90(x, k=1, axes=(2, 3))
        ort_inputs = {session.get_inputs()[0].name: x.astype(np.float32)}
        y = session.run(None, ort_inputs)[0]
        if rotate: y = np.rot90(y, k=3, axes=(2, 3))
        if vflip: y = np.flip(y, axis=2)
        if hflip: y = np.flip(y, axis=3)
        return y

    results = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                results.append(forward_transformed(x, hflip, vflip, rot, session))
    return np.mean(np.stack(results), axis=0)

# -----------------------------
# Inference su singola immagine
# -----------------------------
# -----------------------------
# Inference su singola immagine
# -----------------------------
def enhance_image(img_path, session, output_path, use_self_ensemble=False, luminance_threshold=0.2):
    w, h = 640, 480
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Immagine {img_path} non leggibile.")
        return 0

    img_float = img / 255.0

    # Calcola luminanza (BGR -> RGB)
    Y = 0.2126 * img_float[:,:,2] + 0.7152 * img_float[:,:,1] + 0.0722 * img_float[:,:,0]
    Y_mean = np.mean(Y)
    print(f"Luminanza media di {os.path.basename(img_path)}: {Y_mean:.3f}")

    # 🔹 Se luminanza media sopra soglia → salva immagine così com'è
    if Y_mean > luminance_threshold:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        print(f"⚡ {os.path.basename(img_path)} luminosa → nessun enhancement necessario → salvata così com'è")
        return 0  # tempo = 0 perché non passata al modello

    # Resize e conversione RGB per il modello
    img_resized = cv2.resize(img, (w, h))
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = np.float32(img_resized) / 255.0
    img_tensor = img_tensor.transpose(2, 0, 1)[None, :, :, :]

    # 🔹 Inizio misurazione tempo
    start_time = time.time()

    # Inference ONNX
    if use_self_ensemble:
        enhanced = self_ensemble_numpy(img_tensor, session)
    else:
        ort_inputs = {session.get_inputs()[0].name: img_tensor.astype(np.float32)}
        enhanced = session.run(None, ort_inputs)[0]

    elapsed = time.time() - start_time  # tempo in secondi

    # Postprocessing
    enhanced = enhanced[:, :, :h, :w]
    enhanced = np.clip(enhanced[0].transpose(1, 2, 0), 0, 1)
    enhanced_bgr = cv2.cvtColor(img_as_ubyte(enhanced), cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, enhanced_bgr)
    print(f"✅ {os.path.basename(img_path)} elaborata in {elapsed:.4f} secondi → {output_path}")

    return elapsed


# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Retinexformer - Low-light Image Enhancement (ONNX)")
    parser.add_argument("--input", type=str, required=True, help="Immagine singola o cartella di immagini")
    parser.add_argument("--output", type=str, default="./enhanced", help="Cartella output o immagine output")
    parser.add_argument("--self_ensemble", action="store_true", help="Usa self-ensemble per risultati migliori")
    args = parser.parse_args()

    # 🔹 Caricamento modello ONNX con GPU se disponibile
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(
        "/home/giovannidistasio/vista/models/Retinexformer/pretrained_weights/RetinexFormer.onnx",
        providers=providers
    )
    print(f"✅ Modello ONNX caricato ({session.get_providers()})")

    times = []

    # Singola immagine
    if os.path.isfile(args.input):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        output_path = (
            args.output
            if args.output.endswith(".png") or args.output.endswith(".jpg")
            else os.path.join(args.output, "enhanced1.png")
        )
        t = enhance_image(args.input, session, output_path, args.self_ensemble)
        times.append(t)

    # Cartella di immagini
    elif os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        image_paths = sorted(
            glob(os.path.join(args.input, "*.jpg")) + glob(os.path.join(args.input, "*.png"))
        )
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            output_path = os.path.join(args.output, filename)
            t = enhance_image(img_path, session, output_path, args.self_ensemble)
            times.append(t)

    else:
        print("❌ Errore: percorso input non valido.")
        return

    # 🔹 Calcola media dei tempi (in secondi)
    if len(times) > 1:
        mean_time = np.mean(times)
        print(f"\n📊 Tempo medio inferenza: {mean_time:.4f} secondi ({1/mean_time:.2f} FPS)")

if __name__ == "__main__":
    main()
