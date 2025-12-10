import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL = "distilgpt2"
PROMPT = "def fibonacci(n):"

def run(device, runs=5, max_new_tokens=32):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)

    latencies = []
    for _ in range(runs):
        start = time.time()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=max_new_tokens)
        latencies.append(time.time() - start)
    return sum(latencies) / len(latencies)

def main():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    results = {}
    for device in devices:
        avg = run(device)
        results[device] = avg
        print(f"{device}: {avg:.4f} sec")

if __name__ == "__main__":
    main()
