from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np

model_name = "Qwen/Qwen2.5-7B"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B-AWQ", device_map="auto"
)
model.eval()


def surp_score(text, eps_entropy=2.5, pct_k=40):
    # Tokenize
    ids = tok(text, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(ids, labels=ids)
        # logits: (1, N, V)
        logits = outputs.logits
        # compute probs & log-probs
        logp = torch.nn.functional.log_softmax(logits, dim=-1)
        probs = torch.exp(logp)
    # shift to get next-token predictions
    # P_i predicts x_i given x_{<i}
    P_i = probs[0, :-1, :]
    logpi = logp[0, :-1, ids[0, 1:]]  # shape (N-1,)
    # entropy E_i
    entropy = -(P_i * logp[0, :-1, :]).sum(dim=-1)  # (N-1,)
    # percentile threshold
    Lk = np.percentile(logpi.cpu().numpy(), pct_k)
    # select surprising tokens
    mask = (entropy.cpu().numpy() < eps_entropy) & (logpi.cpu().numpy() < Lk)
    if mask.sum() == 0:
        return float("nan")  # no surprising tokens
    return float(logpi.cpu().numpy()[mask].mean())


def predict(text, threshold):
    score = surp_score(text)
    return "AI-generated" if score >= threshold else "Human-written"


# Example usage
if __name__ == "__main__":
    text = ""
    threshold = -2.0  # Example threshold
    prediction = predict(text, threshold)
    print(f"Prediction: {prediction}")
