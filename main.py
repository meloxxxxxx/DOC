import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig
from datasets import Dataset
from torch.cuda.amp import autocast
from peft import LoraConfig, get_peft_model
from data import *
from pca_utils import *
import pandas as pd
import pyarrow.parquet as pq

def load_lora_model(model_path: str, r=8):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)
    config = AutoConfig.from_pretrained(model_path)

    modules = [name for name, _ in model.named_modules()][1:]
    target_modules = [name for name in modules if "q_proj" in name or "v_proj" in name]
    lora_config = LoraConfig(target_modules=target_modules, r=r)
    peft_model = get_peft_model(model, lora_config)
    peft_model = peft_model.to(torch.bfloat16)
    return peft_model

def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 
    tokenizer.truncation_side = "right" 
    llama2_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + content.strip() }}{% endif %}{% endfor %}"
    tokenizer.chat_template = llama2_template
    return tokenizer

def pca_fine_tune(model, optimizer, data_loader, pca_monitor: pcaMonitor, record_file):
    model.train()
    total_train_loss = 0
    
    for num_batch, batch in enumerate(data_loader):
        if num_batch%10==0:
            print(f'num_batch: {pca_monitor.task_data_n}/{len(data_loader)}')
            print(f'pcamonitor status: {pca_monitor.get_status()}')
            with open(record_file, "a", encoding="utf-8") as file:
                file.write(f'num_batch: {pca_monitor.task_data_n}/{len(data_loader)}\n')
                file.write(f'pcamonitor status: {pca_monitor.get_status()}\n')

        hidden_state_container, hidden_state_hooks = register_hidden_state_hook(model)

        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)
        
        with autocast(dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()

        with torch.inference_mode():
            grads = get_grads(model)
            params = get_parameters(model)
            delta_hs_vector = pca_monitor.compute_delta_hs(params, grads, hidden_state_container)

            pca_monitor.fit(delta_hs_vector)
            
            if pca_monitor.get_status()['task_id'] > 0:
                cut_grads = pca_monitor.cut_components(grads, delta_hs_vector)
                replace_grads(model, cut_grads)

            print(f"residual_rate: {pca_monitor.get_residual_rate()}")
            with open(record_file, "a", encoding="utf-8") as file:
                file.write(f"residual_rate: {pca_monitor.get_residual_rate()}\n")


        with autocast(dtype=torch.bfloat16):
            optimizer.step()
            optimizer.zero_grad()

        for hidden_state_hook in hidden_state_hooks:
            hidden_state_hook.remove()

        torch.cuda.empty_cache()

def test(lr, r, record_file: str):
    torch.cuda.empty_cache()
    model_path = "Llama-2-7b-chat-hf"
    
    model = load_lora_model(model_path, r=r)
    tokenizer = load_tokenizer(model_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #yelp sentiment
    select_seed, train_size= 42, 10000
    table = pq.read_table('datas/yelp_review_full/yelp_review_full/train-00000-of-00001.parquet')
    table = table.to_pandas()
    yelp_train_dataset = Dataset.from_pandas(table).shuffle(seed=select_seed).select(range(train_size))
    yelp_train_loader = load_yelp_data(tokenizer, yelp_train_dataset, max_token = 160, batch_size = 8)

    #yahoo QA
    select_seed, train_size= 42, 10000
    table = pq.read_table('datas/yahoo_answers_topics/yahoo_answers_topics/train-00000-of-00002.parquet')
    table = table.to_pandas()
    yahoo_train_dataset = Dataset.from_pandas(table).shuffle(seed=select_seed).select(range(train_size))
    yahoo_train_loader = load_yahoo_data(tokenizer, yahoo_train_dataset, max_token = 160, batch_size = 8)
    
    #amazon sentiment
    select_seed, train_size= 42, 10000
    df = pd.read_json('datas/amazon_reviews_multi/en/train.jsonl', lines=True)
    amazon_train_dataset = Dataset.from_pandas(df).shuffle(seed=select_seed).select(range(train_size))
    amazon_train_loader = load_amazon_data(tokenizer, amazon_train_dataset, max_token = 160, batch_size = 8)
    
    #agnews topic
    select_seed, train_size= 42, 10000
    table = pq.read_table('datas/ag_news/train-00000-of-00001.parquet')
    table = table.to_pandas()
    agnews_train_dataset = Dataset.from_pandas(table).shuffle(seed=select_seed).select(range(train_size))
    agnews_train_loader = load_agnews_data(tokenizer, agnews_train_dataset, max_token = 160, batch_size = 8)

    #dbpedia topic
    select_seed, train_size= 42, 10000
    table = pq.read_table('datas/dbpedia_14/dbpedia_14/train-00000-of-00001.parquet')
    table = table.to_pandas()
    dbpedia_train_dataset = Dataset.from_pandas(table).shuffle(seed=select_seed).select(range(train_size))
    dbpedia_train_loader = load_dbpedia_data(tokenizer, dbpedia_train_dataset, max_token = 160, batch_size = 8)
    
    loader_list_order1 = [
                    yelp_train_loader, 
                    amazon_train_loader,
                    dbpedia_train_loader,
                    agnews_train_loader, 
                    yahoo_train_loader
                    ]    

    loader_list_order2 = [
                    dbpedia_train_loader,
                    agnews_train_loader, 
                    yelp_train_loader, 
                    amazon_train_loader,
                    yahoo_train_loader
                    ]    

    loader_list_order3 = [
                    yelp_train_loader, 
                    amazon_train_loader,
                    dbpedia_train_loader,
                    agnews_train_loader, 
                    yahoo_train_loader
                    ]    

    pca_monitor = pcaMonitor(k_start=48, k_add=48, drift_eps=0.02, add_eps=0.1, l=2, max_drift_bias=0.1, p=0.5)
    
    for i, fine_tune_loader in enumerate(loader_list_order1):
        pca_fine_tune(model, optimizer, fine_tune_loader, pca_monitor, record_file)
        pca_monitor.new_task()

test(lr=1e-4, r=32, record_file='DOC.txt')
