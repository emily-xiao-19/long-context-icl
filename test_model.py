from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# import pdb; pdb.set_trace()

tokenizer = AutoTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/LLaMA-2-7B-32K", trust_remote_code=True, torch_dtype=torch.float16)
print(model.config)
print(tokenizer)
print("\nModel Parameters Device:")
for name, param in model.named_parameters():
    print(f"{name}: {param.device}")
# Move the model to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

input_context = """query: I've completely forgot the code to get into the app.
intent: passcode forgotten
==
query: Why can't I see my refund in my statement?
intent: Refund not showing up
==
query: for god sake, just delete my account i am sick of this
intent: terminate account
==
query: How do I add money to my card?
intent: topping up by card
==
query: What is the status of my card's delivery?
intent: card arrival
==
query: Am I able to use my account if the identity verification is not complete?
intent: why verify identity
==
query: I need my card as quick as possible
intent: card delivery estimate
==
query: I sent some money but the intended recipient says it hasn't arrived
intent: transfer not received by recipient
==
query: Why does my transfer still say it is pending?
intent: pending transfer
==
query: Which currencies do you accept for adding money?
intent: supported cards and currencies
==
query: I want to add money but don't know what currencies you accept, can you tell me?
intent:"""

input_ids = tokenizer.encode(input_context, return_tensors="pt").to(device)
with torch.no_grad():
    output = model.generate(input_ids, max_length=1000)
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)

# Test with a simple input
simple_input = "Hello, how are you?"
simple_input_ids = tokenizer.encode(simple_input, return_tensors="pt").to(device)

with torch.no_grad():
    simple_output = model.generate(simple_input_ids, max_length=50, temperature=0.7)

simple_output_text = tokenizer.decode(simple_output[0], skip_special_tokens=True)
print("Simple input output:", simple_output_text)
