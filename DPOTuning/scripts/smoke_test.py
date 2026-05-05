import torch
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from alignment import ScriptArguments, SFTConfig, DPOConfig

print("=== DPO-1 Smoke Test ===")
print(f"torch:          {torch.__version__}")
print(f"bitsandbytes:   {bnb.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
print(f"device:         {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
print(f"alignment pkg:  OK ({ScriptArguments.__name__}, {SFTConfig.__name__}, {DPOConfig.__name__} importable)")

# 4-bit smoke test with tiny model
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained("gpt2", quantization_config=bnb_config, device_map="auto")
print(f"4-bit load:     OK | dtype={model.dtype} | device={next(model.parameters()).device}")
print("\nAll checks passed. Ready for DPO-2.")
