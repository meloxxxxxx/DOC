from torch.utils.data import DataLoader
from datasets import Dataset
class LabeledDataset(Dataset):
    def __init__(self, encoded_dataset):
        self.input_ids = encoded_dataset["input_ids"]
        self.attention_mask = encoded_dataset["attention_mask"]
        self.labels = encoded_dataset["labels"]
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }
def load_yelp_data(tokenizer, dataset: Dataset, size = None, seed = None, max_token = 160, batch_size = 16):
    if size is not None:
        dataset = dataset.shuffle(seed=seed).select(range(size))
    def yelp_preprocess_function(examples):
        input_texts = []
        label_texts = []
        for label, text in zip(examples['label'], examples['text']):
            input_messages = [
                {"role": "system", "content": "Score the sentiment of the following comment with numbers 1, 2, 3, 4, 5. Note that the bigger the number, the more positive the comment is."},
                {"role": "user", "content": 'Comment:\n' + text},
                {"role": "assistant", "content": 'The score of the comment is '}  
            ]
            full_messages = [
                {"role": "system", "content": "Score the sentiment of the following comment with numbers 1, 2, 3, 4, 5. Note that the bigger the number, the more positive the comment is."},
                {"role": "user", "content": 'Comment:\n' + text},
                {"role": "assistant", "content": 'The score of the comment is ' + str(int(label)+1)}  
            ]
            full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
            input_text = tokenizer.apply_chat_template(input_messages, tokenize=False)  
            input_texts.append(input_text)
            label_texts.append(full_text)  
        model_inputs = tokenizer(
            input_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        full_encoded = tokenizer(
            label_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        labels = full_encoded["input_ids"]
        input_lengths = [len(seq[seq != tokenizer.pad_token_id]) for seq in model_inputs["input_ids"]]
        for i, input_len in enumerate(input_lengths):
            if input_len < max_token:
                labels[i, :input_len-1] = -100
            if input_len >= max_token:
                labels[i,:] = -100
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels
        }
    encoded_dataset = dataset.map(yelp_preprocess_function, remove_columns=dataset.column_names, batched = True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", 'labels'])
    dataset = LabeledDataset(encoded_dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory = True)
    return loader
def load_yahoo_data(tokenizer, dataset: Dataset, size = None, seed = None, max_token = 160, batch_size = 16):
    if size is not None:
        dataset = dataset.shuffle(seed=seed).select(range(size))
    def yahoo_preprocess_function(examples):
        input_texts = []
        label_texts = []
        for title, question, answer in zip(examples['question_title'], examples['question_content'], examples['best_answer']):
            input_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": title + '\n' + question},
            ]
            full_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": title + '\n' + question},
                {"role": "assistant", "content": answer}  
            ]
            full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
            input_text = tokenizer.apply_chat_template(input_messages, tokenize=False)  
            input_texts.append(input_text)
            label_texts.append(full_text)  
        model_inputs = tokenizer(
            input_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        full_encoded = tokenizer(
            label_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        labels = full_encoded["input_ids"]
        input_lengths = [len(seq[seq != tokenizer.pad_token_id]) for seq in model_inputs["input_ids"]]
        for i, input_len in enumerate(input_lengths):
            if input_len < max_token:
                labels[i, :input_len-1] = -100
            if input_len >= max_token:
                labels[i,:] = -100
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels
        }
    encoded_dataset = dataset.map(yahoo_preprocess_function, remove_columns=dataset.column_names, batched = True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", 'labels'])
    dataset = LabeledDataset(encoded_dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory = True)
    return loader
def load_amazon_data(tokenizer, dataset: Dataset, size = None, seed = None, max_token = 160, batch_size = 16):
    if size is not None:
        dataset = dataset.shuffle(seed=seed).select(range(size))
    def amazon_preprocess_function(examples):
        input_texts = []
        label_texts = []
        for label, text in zip(examples['label'], examples['text']):
            input_messages = [
                {"role": "system", "content": "Score the sentiment of the following comment with numbers 1, 2, 3, 4, 5. Note that the bigger the number, the more positive the comment is."},
                {"role": "user", "content": 'Comment:\n' + text},
                {"role": "assistant", "content": 'The score of the comment is '}  
            ]
            full_messages = [
                {"role": "system", "content": "Score the sentiment of the following comment with numbers 1, 2, 3, 4, 5. Note that the bigger the number, the more positive the comment is."},
                {"role": "user", "content": 'Comment:\n' + text},
                {"role": "assistant", "content": 'The score of the comment is ' + str(int(label)+1)}  
            ]
            full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
            input_text = tokenizer.apply_chat_template(input_messages, tokenize=False)  
            input_texts.append(input_text)
            label_texts.append(full_text)  
        model_inputs = tokenizer(
            input_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        full_encoded = tokenizer(
            label_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        labels = full_encoded["input_ids"]
        input_lengths = [len(seq[seq != tokenizer.pad_token_id]) for seq in model_inputs["input_ids"]]
        for i, input_len in enumerate(input_lengths):
            if input_len < max_token:
                labels[i, :input_len-1] = -100
            if input_len >= max_token:
                labels[i,:] = -100
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels
        }
    encoded_dataset = dataset.map(amazon_preprocess_function, remove_columns=dataset.column_names, batched = True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", 'labels'])
    dataset = LabeledDataset(encoded_dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory = True)
    return loader
def load_agnews_data(tokenizer, dataset: Dataset, size = None, seed = None, max_token = 160, batch_size = 16):
    if size is not None:
        dataset = dataset.shuffle(seed=seed).select(range(size))
    def agnews_preprocess_function(examples):
        label_dict = {'0': 'World', '1': 'Sports', '2': 'Business', '3': 'Science and technology'}
        input_texts = []
        label_texts = []
        for label, text in zip(examples['label'], examples['text']):
            input_messages = [
                {"role": "system", "content": "What is the topic of the following paragraph? Choose one from the option. Options are: World, Sports, Business, Science and technology."},
                {"role": "user", "content": 'Paragraph:\n' + text},
                {"role": "assistant", "content": 'The topic is '}  
            ]
            full_messages = [
                {"role": "system", "content": "What is the topic of the following paragraph? Choose one from the option. Options are: World, Sports, Business, Science and technology."},
                {"role": "user", "content": 'Paragraph:\n' + text},
                {"role": "assistant", "content": 'The topic is ' + label_dict[str(label)] + "."}  
            ]
            full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
            input_text = tokenizer.apply_chat_template(input_messages, tokenize=False)  
            input_texts.append(input_text)
            label_texts.append(full_text)  
        model_inputs = tokenizer(
            input_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        full_encoded = tokenizer(
            label_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        labels = full_encoded["input_ids"]
        input_lengths = [len(seq[seq != tokenizer.pad_token_id]) for seq in model_inputs["input_ids"]]
        for i, input_len in enumerate(input_lengths):
            if input_len < max_token:
                labels[i, :input_len-1] = -100
            if input_len >= max_token:
                labels[i,:] = -100
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels
        }
    encoded_dataset = dataset.map(agnews_preprocess_function, remove_columns=dataset.column_names, batched = True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", 'labels'])
    dataset = LabeledDataset(encoded_dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory = True)
    return loader
def load_dbpedia_data(tokenizer, dataset: Dataset, size = None, seed = None, max_token = 160, batch_size = 16):
    if size is not None:
        dataset = dataset.shuffle(seed=seed).select(range(size))
    def dbpedia_preprocess_function(examples):
        label_dict = {'0': 'Company',
            '1': 'EducationalInstitution',
            '2': 'Artist',
            '3': 'Athlete',
            '4': 'OfficeHolder',
            '5': 'MeanOfTransportation',
            '6': 'Building',
            '7': 'NaturalPlace',
            '8': 'Village',
            '9': 'Animal',
            '10': 'Plant',
            '11': 'Album',
            '12': 'Film',
            '13': 'WrittenWork'}
        input_texts = []
        label_texts = []
        for label, title, content in zip(examples['label'], examples['title'], examples['content']):
            input_messages = [
                {"role": "system", "content": "What is the topic of the following paragraph? Choose one from the option. Options are: Company, EducationalInstitution,, Artist, Athlete, OfficeHolder, MeanOfTransportation, Building, NaturalPlace, Village, Animal, Plant, Album, Film, WrittenWork."},
                {"role": "user", "content": 'Paragraph:\n' + title + '\n' + content},
                {"role": "assistant", "content": 'The topic is '}  
            ]
            full_messages = [
                {"role": "system", "content": "What is the topic of the following paragraph? Choose one from the option. Options are: Company, EducationalInstitution,, Artist, Athlete, OfficeHolder, MeanOfTransportation, Building, NaturalPlace, Village, Animal, Plant, Album, Film, WrittenWork."},
                {"role": "user", "content": 'Paragraph:\n' + title + '\n' + content},
                {"role": "assistant", "content": 'The topic is ' + label_dict[str(label)] + "."}  
            ]
            full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
            input_text = tokenizer.apply_chat_template(input_messages, tokenize=False)  
            input_texts.append(input_text)
            label_texts.append(full_text)  
        model_inputs = tokenizer(
            input_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        full_encoded = tokenizer(
            label_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        labels = full_encoded["input_ids"]
        input_lengths = [len(seq[seq != tokenizer.pad_token_id]) for seq in model_inputs["input_ids"]]
        for i, input_len in enumerate(input_lengths):
            if input_len < max_token:
                labels[i, :input_len-1] = -100
            if input_len >= max_token:
                labels[i,:] = -100
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels
        }
    encoded_dataset = dataset.map(dbpedia_preprocess_function, remove_columns=dataset.column_names, batched = True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", 'labels'])
    dataset = LabeledDataset(encoded_dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory = True)
    return loader
def load_mnli_data(tokenizer, dataset: Dataset, size = None, seed = None, max_token = 160, batch_size = 16):
    if size is not None:
        dataset = dataset.shuffle(seed=seed).select(range(size))
    def mnli_preprocess_function(examples):
        label_dict = {'0': 'entailment',
            '1': 'neutral',
            '2': 'contradiction'}
        input_texts = []
        label_texts = []
        for text1, text2, label in zip(examples['text1'], examples['text2'], examples['label']):
            input_messages = [
                {"role": "system", "content": "What is the logical relationship between the 'sentence 1' and the 'sentence 2'? Choose one from the option. Options are: entailment, neutral, contradiction."},
                {"role": "user", "content": 'sentence 1:\n' + text1 + '\n' + 'sentence 2:\n' + text2},
                {"role": "assistant", "content": 'The relationship is '}  
            ]
            full_messages = [
                {"role": "system", "content": "What is the logical relationship between the 'sentence 1' and the 'sentence 2'? Choose one from the option. Options are: entailment, neutral, contradiction."},
                {"role": "user", "content": 'sentence 1:\n' + text1 + '\n' + 'sentence 2:\n' + text2},
                {"role": "assistant", "content": 'The relationship is ' + label_dict[str(label)] + "."}  
            ]
            full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
            input_text = tokenizer.apply_chat_template(input_messages, tokenize=False)  
            input_texts.append(input_text)
            label_texts.append(full_text)  
        model_inputs = tokenizer(
            input_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        full_encoded = tokenizer(
            label_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        labels = full_encoded["input_ids"]
        input_lengths = [len(seq[seq != tokenizer.pad_token_id]) for seq in model_inputs["input_ids"]]
        for i, input_len in enumerate(input_lengths):
            if input_len < max_token:
                labels[i, :input_len-1] = -100
            if input_len >= max_token:
                labels[i,:] = -100
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels
        }
    encoded_dataset = dataset.map(mnli_preprocess_function, remove_columns=dataset.column_names, batched = True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", 'labels'])
    dataset = LabeledDataset(encoded_dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory = True)
    return loader
def load_rte_data(tokenizer, dataset: Dataset, size = None, seed = None, max_token = 160, batch_size = 16):
    if size is not None:
        dataset = dataset.shuffle(seed=seed).select(range(size))
    def rte_preprocess_function(examples):
        label_dict = {'0': 'entailment',
            '1': 'not entailment'}
        input_texts = []
        label_texts = []
        for text1, text2, label in zip(examples['text1'], examples['text2'], examples['label']):
            input_messages = [
                {"role": "system", "content": "What is the logical relationship between the 'sentence 1' and the 'sentence 2'? Choose one from the option. Options are: entailment, not entailment."},
                {"role": "user", "content": 'sentence 1:\n' + text1 + '\n' + 'sentence 2:\n' + text2},
                {"role": "assistant", "content": 'The relationship is '}  
            ]
            full_messages = [
                {"role": "system", "content": "What is the logical relationship between the 'sentence 1' and the 'sentence 2'? Choose one from the option. Options are: entailment, not entailment."},
                {"role": "user", "content": 'sentence 1:\n' + text1 + '\n' + 'sentence 2:\n' + text2},
                {"role": "assistant", "content": 'The relationship is ' + label_dict[str(label)] + "."}  
            ]
            full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
            input_text = tokenizer.apply_chat_template(input_messages, tokenize=False)  
            input_texts.append(input_text)
            label_texts.append(full_text)  
        model_inputs = tokenizer(
            input_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        full_encoded = tokenizer(
            label_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        labels = full_encoded["input_ids"]
        input_lengths = [len(seq[seq != tokenizer.pad_token_id]) for seq in model_inputs["input_ids"]]
        for i, input_len in enumerate(input_lengths):
            if input_len < max_token:
                labels[i, :input_len-1] = -100
            if input_len >= max_token:
                labels[i,:] = -100
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels
        }
    encoded_dataset = dataset.map(rte_preprocess_function, remove_columns=dataset.column_names, batched = True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", 'labels'])
    dataset = LabeledDataset(encoded_dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory = True)
    return loader
def load_sst2_data(tokenizer, dataset: Dataset, size = None, seed = None, max_token = 160, batch_size = 16):
    if size is not None:
        dataset = dataset.shuffle(seed=seed).select(range(size))
    def sst2_preprocess_function(examples):
        input_texts = []
        label_texts = []
        answer_dict = {'0': 'positive', '1': 'negative'}
        for label, text in zip(examples['label'], examples['text']):
            input_messages = [
                {"role": "system", "content": "What is the sentiment of the following paragraph? Choose one from the option. Options are: positive, negative."},
                {"role": "user", "content": 'Paragraph:\n' + text},
                {"role": "assistant", "content": 'The sentiment of the paragraph is '}  
            ]
            full_messages = [
                {"role": "system", "content": "What is the sentiment of the following paragraph? Choose one from the option. Options are: positive, negative."},
                {"role": "user", "content": 'Paragraph:\n' + text},
                {"role": "assistant", "content": 'The sentiment of the paragraph is ' + answer_dict[str(label)]}  
            ]
            full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
            input_text = tokenizer.apply_chat_template(input_messages, tokenize=False)  
            input_texts.append(input_text)
            label_texts.append(full_text)  
        model_inputs = tokenizer(
            input_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        full_encoded = tokenizer(
            label_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        labels = full_encoded["input_ids"]
        input_lengths = [len(seq[seq != tokenizer.pad_token_id]) for seq in model_inputs["input_ids"]]
        for i, input_len in enumerate(input_lengths):
            if input_len < max_token:
                labels[i, :input_len-1] = -100
            if input_len >= max_token:
                labels[i,:] = -100
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels
        }
    encoded_dataset = dataset.map(sst2_preprocess_function, remove_columns=dataset.column_names, batched = True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", 'labels'])
    dataset = LabeledDataset(encoded_dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory = True)
    return loader
def load_imdb_data(tokenizer, dataset: Dataset, size = None, seed = None, max_token = 160, batch_size = 16):
    if size is not None:
        dataset = dataset.shuffle(seed=seed).select(range(size))
    def imdb_preprocess_function(examples):
        input_texts = []
        label_texts = []
        answer_dict = {'0': 'negative', '1': 'positive'}
        for label, text in zip(examples['label'], examples['text']):
            input_messages = [
                {"role": "system", "content": "What is the sentiment of the following paragraph? Choose one from the option. Options are: positive, negative."},
                {"role": "user", "content": 'Paragraph:\n' + text},
                {"role": "assistant", "content": 'The sentiment of the paragraph is '}  
            ]
            full_messages = [
                {"role": "system", "content": "What is the sentiment of the following paragraph? Choose one from the option. Options are: positive, negative."},
                {"role": "user", "content": 'Paragraph:\n' + text},
                {"role": "assistant", "content": 'The sentiment of the paragraph is ' + answer_dict[str(label)]}  
            ]
            full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
            input_text = tokenizer.apply_chat_template(input_messages, tokenize=False)  
            input_texts.append(input_text)
            label_texts.append(full_text)  
        model_inputs = tokenizer(
            input_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        full_encoded = tokenizer(
            label_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        labels = full_encoded["input_ids"]
        input_lengths = [len(seq[seq != tokenizer.pad_token_id]) for seq in model_inputs["input_ids"]]
        for i, input_len in enumerate(input_lengths):
            if input_len < max_token:
                labels[i, :input_len-1] = -100
            if input_len >= max_token:
                labels[i,:] = -100
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels
        }
    encoded_dataset = dataset.map(imdb_preprocess_function, remove_columns=dataset.column_names, batched = True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", 'labels'])
    dataset = LabeledDataset(encoded_dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory = True)
    return loader
def load_qqp_data(tokenizer, dataset: Dataset, size = None, seed = None, max_token = 160, batch_size = 16):
    if size is not None:
        dataset = dataset.shuffle(seed=seed).select(range(size))
    def qqp_preprocess_function(examples):
        label_dict = {'0': 'no',
            '1': 'yes'}
        input_texts = []
        label_texts = []
        for text1, text2, label in zip(examples['question1'], examples['question2'], examples['label']):
            input_messages = [
                {"role": "system", "content": "Whether the 'sentence 1' and the 'sentence 2' have the same meaning? Choose one from the option. Options are: yes, no."},
                {"role": "user", "content": 'sentence 1:\n' + text1 + '\n' + 'sentence 2:\n' + text2},
                {"role": "assistant", "content": 'The answer is '}  
            ]
            full_messages = [
                {"role": "system", "content": "Whether the 'sentence 1' and the 'sentence 2' have the same meaning? Choose one from the option. Options are: yes, no."},
                {"role": "user", "content": 'sentence 1:\n' + text1 + '\n' + 'sentence 2:\n' + text2},
                {"role": "assistant", "content": 'The answer is ' + label_dict[str(label)] + "."}  
            ]
            full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
            input_text = tokenizer.apply_chat_template(input_messages, tokenize=False)  
            input_texts.append(input_text)
            label_texts.append(full_text)  
        model_inputs = tokenizer(
            input_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        full_encoded = tokenizer(
            label_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        labels = full_encoded["input_ids"]
        input_lengths = [len(seq[seq != tokenizer.pad_token_id]) for seq in model_inputs["input_ids"]]
        for i, input_len in enumerate(input_lengths):
            if input_len < max_token:
                labels[i, :input_len-1] = -100
            if input_len >= max_token:
                labels[i,:] = -100
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels
        }
    encoded_dataset = dataset.map(qqp_preprocess_function, remove_columns=dataset.column_names, batched = True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", 'labels'])
    dataset = LabeledDataset(encoded_dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory = True)
    return loader
def load_wic_data(tokenizer, dataset: Dataset, size = None, seed = None, max_token = 160, batch_size = 16):
    if size is not None:
        dataset = dataset.shuffle(seed=seed).select(range(size))
    def wic_preprocess_function(examples):
        label_dict = {'0': 'entailment',
            '1': 'not entailment'}
        input_texts = []
        label_texts = []
        for sentence, label in zip(examples['sentence'], examples['label']):
            input_messages = [
                {"role": "system", "content": "Given a word and two sentences, whether the word is used with the same sense in both sentence? Choose one from the option. Options are: True, False"},
                {"role": "user", "content": sentence},
                {"role": "assistant", "content": 'The answer is '}  
            ]
            full_messages = [
                {"role": "system", "content": "Given a word and two sentences, whether the word is used with the same sense in both sentence? Choose one from the option. Options are: True, False"},
                {"role": "user", "content": sentence},
                {"role": "assistant", "content": 'The answer is '+ label}  
            ]
            full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
            input_text = tokenizer.apply_chat_template(input_messages, tokenize=False)  
            input_texts.append(input_text)
            label_texts.append(full_text)  
        model_inputs = tokenizer(
            input_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        full_encoded = tokenizer(
            label_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        labels = full_encoded["input_ids"]
        input_lengths = [len(seq[seq != tokenizer.pad_token_id]) for seq in model_inputs["input_ids"]]
        for i, input_len in enumerate(input_lengths):
            if input_len < max_token:
                labels[i, :input_len-1] = -100
            if input_len >= max_token:
                labels[i,:] = -100
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels
        }
    encoded_dataset = dataset.map(wic_preprocess_function, remove_columns=dataset.column_names, batched = True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", 'labels'])
    dataset = LabeledDataset(encoded_dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory = True)
    return loader
def load_cb_data(tokenizer, dataset: Dataset, size = None, seed = None, max_token = 160, batch_size = 16):
    if size is not None:
        dataset = dataset.shuffle(seed=seed).select(range(size))
    def cb_preprocess_function(examples):
        input_texts = []
        label_texts = []
        for sentence, label in zip(examples['sentence'], examples['label']):
            input_messages = [
                {"role": "system", "content": "What is the logical relationship between the 'sentence 1' and the 'sentence 2'? Choose one from the option. Options are: entailment, contradiction, neutral."},
                {"role": "user", "content": sentence},
                {"role": "assistant", "content": 'The relationship is '}  
            ]
            full_messages = [
                {"role": "system", "content": "What is the logical relationship between the 'sentence 1' and the 'sentence 2'? Choose one from the option. Options are: entailment, contradiction, neutral."},
                {"role": "user", "content": sentence},
                {"role": "assistant", "content": 'The relationship is ' + label + "."}  
            ]
            full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
            input_text = tokenizer.apply_chat_template(input_messages, tokenize=False)  
            input_texts.append(input_text)
            label_texts.append(full_text)  
        model_inputs = tokenizer(
            input_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        full_encoded = tokenizer(
            label_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        labels = full_encoded["input_ids"]
        input_lengths = [len(seq[seq != tokenizer.pad_token_id]) for seq in model_inputs["input_ids"]]
        for i, input_len in enumerate(input_lengths):
            if input_len < max_token:
                labels[i, :input_len-1] = -100
            if input_len >= max_token:
                labels[i,:] = -100
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels
        }
    encoded_dataset = dataset.map(cb_preprocess_function, remove_columns=dataset.column_names, batched = True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", 'labels'])
    dataset = LabeledDataset(encoded_dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory = True)
    return loader
def load_copa_data(tokenizer, dataset: Dataset, size = None, seed = None, max_token = 160, batch_size = 16):
    if size is not None:
        dataset = dataset.shuffle(seed=seed).select(range(size))
    def copa_preprocess_function(examples):
        input_texts = []
        label_texts = []
        for sentence, label in zip(examples['sentence'], examples['label']):
            input_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": sentence},
                {"role": "assistant", "content": 'The answer is '}  
            ]
            full_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": sentence},
                {"role": "assistant", "content": 'The answer is ' + label + '.'}  
            ]
            full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
            input_text = tokenizer.apply_chat_template(input_messages, tokenize=False)  
            input_texts.append(input_text)
            label_texts.append(full_text)  
        model_inputs = tokenizer(
            input_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        full_encoded = tokenizer(
            label_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        labels = full_encoded["input_ids"]
        input_lengths = [len(seq[seq != tokenizer.pad_token_id]) for seq in model_inputs["input_ids"]]
        for i, input_len in enumerate(input_lengths):
            if input_len < max_token:
                labels[i, :input_len-1] = -100
            if input_len >= max_token:
                labels[i,:] = -100
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels
        }
    encoded_dataset = dataset.map(copa_preprocess_function, remove_columns=dataset.column_names, batched = True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", 'labels'])
    dataset = LabeledDataset(encoded_dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory = True)
    return loader
def load_boolqa_data(tokenizer, dataset: Dataset, size = None, seed = None, max_token = 160, batch_size = 16):
    if size is not None:
        dataset = dataset.shuffle(seed=seed).select(range(size))
    def boolqa_preprocess_function(examples):
        input_texts = []
        label_texts = []
        for sentence, label in zip(examples['sentence'], examples['label']):
            input_messages = [
                {"role": "system", "content": "According to the following passage, is the question true or false? Choose one from the option. Options are: True, False."},
                {"role": "user", "content": sentence},
                {"role": "assistant", "content": 'The answer is '}  
            ]
            full_messages = [
                {"role": "system", "content": "According to the following passage, is the question true or false? Choose one from the option. Options are: True, False."},
                {"role": "user", "content": sentence},
                {"role": "assistant", "content": 'The answer is ' + label + "."}  
            ]
            full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
            input_text = tokenizer.apply_chat_template(input_messages, tokenize=False)  
            input_texts.append(input_text)
            label_texts.append(full_text)  
        model_inputs = tokenizer(
            input_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        full_encoded = tokenizer(
            label_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        labels = full_encoded["input_ids"]
        input_lengths = [len(seq[seq != tokenizer.pad_token_id]) for seq in model_inputs["input_ids"]]
        for i, input_len in enumerate(input_lengths):
            if input_len < max_token:
                labels[i, :input_len-1] = -100
            if input_len >= max_token:
                labels[i,:] = -100
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels
        }
    encoded_dataset = dataset.map(boolqa_preprocess_function, remove_columns=dataset.column_names, batched = True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", 'labels'])
    dataset = LabeledDataset(encoded_dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory = True)
    return loader
def load_multirc_data(tokenizer, dataset: Dataset, size = None, seed = None, max_token = 160, batch_size = 16):
    if size is not None:
        dataset = dataset.shuffle(seed=seed).select(range(size))
    def multirc_preprocess_function(examples):
        input_texts = []
        label_texts = []
        for sentence, label in zip(examples['sentence'], examples['label']):
            input_messages = [
                {"role": "system", "content": "According to the following passage and question, is the candidate answer true or false? Choose one from the option. Options are: True, False."},                
                {"role": "user", "content": sentence},
                {"role": "assistant", "content": 'The answer is '}  
            ]
            full_messages = [
                {"role": "system", "content": "According to the following passage and question, is the candidate answer true or false? Choose one from the option. Options are: True, False."},
                {"role": "user", "content": sentence},
                {"role": "assistant", "content": 'The answer is ' + label + "."}  
            ]
            full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
            input_text = tokenizer.apply_chat_template(input_messages, tokenize=False)  
            input_texts.append(input_text)
            label_texts.append(full_text)  
        model_inputs = tokenizer(
            input_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        full_encoded = tokenizer(
            label_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="right"
        )
        labels = full_encoded["input_ids"]
        input_lengths = [len(seq[seq != tokenizer.pad_token_id]) for seq in model_inputs["input_ids"]]
        for i, input_len in enumerate(input_lengths):
            if input_len < max_token:
                labels[i, :input_len-1] = -100
            if input_len >= max_token:
                labels[i,:] = -100
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels
        }
    encoded_dataset = dataset.map(multirc_preprocess_function, remove_columns=dataset.column_names, batched = True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", 'labels'])
    dataset = LabeledDataset(encoded_dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory = True)
    return loader