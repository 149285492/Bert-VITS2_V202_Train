# 注释掉日语导入
from text import chinese, english, cleaned_text_to_sequence
# from text import chinese, japanese, english, cleaned_text_to_sequence


# 移除日语模块映射
language_module_map = {"ZH": chinese, "EN": english}
# language_module_map = {"ZH": chinese, "JP": japanese, "EN": english}


def clean_text(text, language):
    # 如果请求日语支持，返回中文处理结果
    if language == "JP":
        language = "ZH"  # 将日语请求重定向到中文处理
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph


def clean_text_bert(text, language):
    # 如果请求日语支持，返回中文处理结果
    if language == "JP":
        language = "ZH"  # 将日语请求重定向到中文处理
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    bert = language_module.get_bert_feature(norm_text, word2ph)
    return phones, tones, bert


def text_to_sequence(text, language):
    # 如果请求日语支持，返回中文处理结果
    if language == "JP":
        language = "ZH"  # 将日语请求重定向到中文处理
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == "__main__":
    pass