import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import numpy as np
import pickle
from src.DetectLM import DetectLM
from src.PerplexityEvaluator import PerplexityEvaluator
from src.PrepareArticles import PrepareArticles #Idan 
from src.fit_survival_function import fit_per_length_survival_function
from glob import glob
import spacy
import re


logging.basicConfig(level=logging.INFO)


def read_all_csv_files(pattern):
    df = pd.DataFrame()
    print(pattern)
    for f in glob(pattern):
        df = pd.concat([df, pd.read_csv(f)])
    return df


def get_survival_function(df, G=101):
    """
    Returns a survival function for every sentence length in tokens.

    Args:
    :df:  data frame with columns 'response' and 'length'
    :G:   number of interpolation points
    
    Return:
        bivariate function (length, responce) -> (0,1)

    """
    assert not df.empty
    value_name = "response" if "response" in df.columns else "logloss"

    df1 = df[~df[value_name].isna()]
    ll = df1['length']
    xx1 = df1[value_name]
    return fit_per_length_survival_function(ll, xx1, log_space=True, G=G)


def mark_edits_remove_tags(chunks, tag="edit"):
    text_chunks = chunks['text']
    edits = []
    for i,text in enumerate(text_chunks):
        chunk_text = re.findall(rf"<{tag}>(.+)</{tag}>", text)
        if len(chunk_text) > 0:
            import pdb; pdb.set_trace()
            chunks['text'][i] = chunk_text[0]
            chunks['length'][i] -= 2
            edits.append(True)
        else:
            edits.append(False)

    return chunks, edits

def get_null_data(model_name, topic):
    data = None
    try:
        file = open(f'nullData/{model_name}_{topic}.pkl', 'rb')
        data = pickle.load(file)
    except:
        pass

    return data

def get_threshold_obj(model_name, topic):
    threshold = None
    try:
        file = open('threshold_obj.pkl', 'rb')
        threshold_obj = pickle.load(file)
        threshold = threshold_obj[model_name][topic]
    except:
        pass

    return threshold

def detect_human_text(model_name, topic, text):
    
    # Get null data
    print('Get null data')
    df_null = get_null_data(model_name, topic)
    if 'num' in df_null.columns:
        df_null = df_null[df_null.num > 1]
    
    # Get survival function
        print('Get survival function')
    pval_functions = get_survival_function(df_null, G=43)

    min_tokens_per_sentence = 10
    max_tokens_per_sentence = 100

    # Init model
    print('Init model')
    lm_name = 'gpt2-xl' if model_name == 'GPT2XL' else 'microsoft/phi-2'
    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    model = AutoModelForCausalLM.from_pretrained(lm_name)
    
    print('Init PerplexityEvaluator')
    sentence_detector = PerplexityEvaluator(model, tokenizer)

    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(f'device {device}')
    model.to(device)

    print('Init DetectLM')
    detector = DetectLM(sentence_detector, pval_functions,
                        min_len=min_tokens_per_sentence,
                        max_len=max_tokens_per_sentence,
                        length_limit_policy='truncate',
                        HC_type='stbl',
                        ignore_first_sentence= False
                        )

    # Convert text to object
    print('Analyze text')
    article_obj = get_article_obj(text)
    parser = PrepareArticles(article_obj, min_tokens=min_tokens_per_sentence, max_tokens=max_tokens_per_sentence)
    chunks = parser(combined=False)

    # Go over all the document
    for i in range(len(chunks['text'])):
        print(chunks['text'][i])
        # for p,v in enumerate(chunks['text'][i]):
        #     print(f'{p}: {v}')
        res = detector(chunks['text'][i], chunks['context'][i], dashboard=None)

        # print(f"Num of Edits (rate) = {np.sum(df['tag'] == '<edit>')} ({edit_rate})")
        # print(f"HC = {res['HC']}")
        # print(f"Fisher = {res['fisher']}")
        # print(f"Fisher (chisquared pvalue) = {res['fisher_pvalue']}")

        results = res['HC']
    
    threshold = get_threshold_obj(model_name, topic)
    print(f"threshold: {threshold}, results: {results}")
    return '1' if results >= threshold else '0'

# Convert article text into object
def get_article_obj(text):
    # Init article object
    article_obj = {
        'sub_titles': [{
            'sentences': []
        }]
    }
    
    nlp = spacy.load("en_core_web_sm")  # Load model

    for line in text.split('\n'):
        doc = nlp(line) # Analyze text
        sentences = [sent.text for sent in doc.sents if len(sent) >= 10] # Split it by sentence
        for sentence in sentences:
            sentence = re.sub(r' +', ' ', sentence) # Remove duplicate spaces
            article_obj['sub_titles'][0]['sentences'].append({'sentence': sentence})

    return article_obj
