from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import torch

dataset = load_dataset("squad")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
valid_dataset = dataset["validation"].shuffle(seed=42).select(range(1000))

def preprocess_function(examples):
    inputs = [f"Generate Question: {context}" for context in examples["context"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["question"], max_length=32, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
    

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["context", "question", "id"])
tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True, remove_columns=["context", "question", "id"])
model = T5ForConditionalGeneration.from_pretrained("t5-small")
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    report_to="none"
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,
    tokenizer=tokenizer,
)

trainer.train()

def generate_question(context):
    input_text = f"Generate Question: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

context = str(input("Enter Text: "))
generated_question = generate_question(context)
print(f"Generated Question: {generated_question}")
