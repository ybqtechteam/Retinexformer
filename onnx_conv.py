import torch
from basicsr.models import create_model
from basicsr.utils.options import parse
import os

# --- Configurazione ---
opt_file = "Options/RetinexFormer_LOL_v2_real.yml"
weights_file = "pretrained_weights/LOL_v2_real.pth"
onnx_file = "RetinexFormer.onnx"

# --- Parsing opzioni ---
opt = parse(opt_file, is_train=False)
opt['dist'] = False  # disabilita distributed training

# --- Creazione modello ---
model = create_model(opt).net_g

# --- Caricamento pesi ---
checkpoint = torch.load(weights_file, map_location="cpu")  # usa CPU per evitare OOM
if 'params' in checkpoint:
    state_dict = checkpoint['params']
else:
    state_dict = checkpoint
try:
    model.load_state_dict(state_dict, strict=False)
except RuntimeError:
    # aggiusta prefisso "module." se presente
    new_state_dict = {'module.'+k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)

# --- Imposta modello in eval mode ---
model.eval()
model.cuda()  # forza CPU per esportazione

# --- Dummy input per ONNX ---
dummy_input = torch.rand(1, 3, 480, 640, dtype=torch.float32)  # risoluzione ridotta per sicurezza
dummy_input = dummy_input.cuda()

# --- Esporta in ONNX ---
torch.onnx.export(
    model,
    dummy_input,
    onnx_file,
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    do_constant_folding=True
)

print(f"✅ Modello esportato correttamente in {onnx_file}")
