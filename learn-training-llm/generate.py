from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("./tiny-model")
tokenizer = GPT2Tokenizer.from_pretrained("./tiny-model")

prompt = "I love"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

output = model.generate(input_ids, max_length=30, num_return_sequences=1)
print(tokenizer.decode(output[0], skip_special_tokens=True))
