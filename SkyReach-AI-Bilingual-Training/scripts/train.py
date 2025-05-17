import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

def main():
    with open("train_config.yaml") as f:
        config = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])
    model = AutoModelForCausalLM.from_pretrained(config['model_name'])

    dataset = load_dataset('json', data_files=config['dataset_path'])['train']

    def tokenize_fn(example):
        return tokenizer(example['text'], truncation=True, max_length=512)

    tokenized_ds = dataset.map(tokenize_fn, batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        logging_dir='./logs',
        logging_steps=10,
        save_steps=100,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds
    )

    trainer.train()

if __name__ == "__main__":
    main()
