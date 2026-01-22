import sys

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from config import config
# from text.japanese import text2sep_kata  # 已经注释掉，暂时不使用

LOCAL_PATH = "./bert/deberta-v2-large-japanese"

tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)

models = dict()


def get_bert_feature(text, word2ph, device=config.bert_gen_config.device):
    # 由于已经注释掉日语支持，返回空张量作为占位符
    # sep_text, _, _ = text2sep_kata(text)
    # sep_tokens = [tokenizer.tokenize(t) for t in sep_text]
    # sep_ids = [tokenizer.convert_tokens_to_ids(t) for t in sep_tokens]
    # sep_ids = [2] + [item for sublist in sep_ids for item in sublist] + [3]
    # return get_bert_feature_with_token(sep_ids, word2ph, device)
    
    # 返回占位符张量，形状与预期相符
    import torch
    return torch.zeros((1024, len(word2ph)-2))  # 假设模型输出维度是1024


def get_bert_feature_with_token(tokens, word2ph, device=config.bert_gen_config.device):
    # 由于已经注释掉日语支持，返回空张量作为占位符
    # if (
    #     sys.platform == "darwin"
    #     and torch.backends.mps.is_available()
    #     and device == "cpu"
    # ):
    #     device = "mps"
    # if not device:
    #     device = "cuda"
    # if device not in models.keys():
    #         models[device] = AutoModelForMaskedLM.from_pretrained(LOCAL_PATH).to(device)
    # with torch.no_grad():
    #     inputs = torch.tensor(tokens).to(device).unsqueeze(0)
    #     token_type_ids = torch.zeros_like(inputs).to(device)
    #     attention_mask = torch.ones_like(inputs).to(device)
    #     inputs = {
    #         "input_ids": inputs,
    #         "token_type_ids": token_type_ids,
    #         "attention_mask": attention_mask,
    #     }

    #     # for i in inputs:
    #     #     inputs[i] = inputs[i].to(device)
    #     res = models[device](**inputs, output_hidden_states=True)
    #     res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
    # assert inputs["input_ids"].shape[-1] == len(word2ph)
    # word2phone = word2ph
    # phone_level_feature = []
    # for i in range(len(word2phone)):
    #     repeat_feature = res[i].repeat(word2phone[i], 1)
    #     phone_level_feature.append(repeat_feature)

    # phone_level_feature = torch.cat(phone_level_feature, dim=0)

    # return phone_level_feature.T
    
    # 返回占位符张量，形状与预期相符
    import torch
    return torch.zeros((1024, len(word2ph)-2))  # 假设模型输出维度是1024