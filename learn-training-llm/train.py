from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# Load tiny GPT-2
model_name = "sshleifer/tiny-gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Load your tiny dataset
dataset = load_dataset("text", data_files={"train": "train.txt"})

# Tokenize
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=32)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Training setup
training_args = TrainingArguments(
    output_dir="./tiny-model",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    logging_steps=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

trainer.train()

# Save model
trainer.save_model("./tiny-model")
tokenizer.save_pretrained("./tiny-model")
