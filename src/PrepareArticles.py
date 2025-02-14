import pandas as pd
import numpy as np
import json

class PrepareArticles(object):
    """
    Parse preprocessed data from csv

    This information is needed for evaluating log-perplexity of the text with respect to a language model
    and later on to test the likelihood that the sentence was sampled from the model with the relevant context.
    """
    def __init__(self, article_obj, get_edits=False, min_tokens=10, max_tokens=100, max_sentences=None):
        self.article_obj = article_obj
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.get_edits = get_edits
        self.max_sentences = max_sentences

    def __call__(self, combined=True):
        return self.parse_dataset(combined)
    
    def parse_dataset(self, combined=True):

        texts = []
        lengths = []
        contexts = []
        tags = []
        
        current_texts = []
        current_lengths = []
        current_contexts = []
        current_tags = []
        exceeded_max_sentences = False
        
        for sub_title in self.article_obj['sub_titles']: # For each sub title
            for sentence in sub_title['sentences']: # Go over each sentence
                sentence_size = len(sentence['sentence'].split())
                if sentence_size >= self.min_tokens and sentence_size <= self.max_tokens:
                    current_texts.append(sentence['sentence'])
                    current_lengths.append(len(sentence['sentence'].split())) # Number of tokens
                    current_contexts.append(sentence['context'] if 'context' in sentence else None)
                    current_tags.append('no edits')

                # If get_edits and has edited sentence save it
                if self.get_edits and 'alternative' in sentence and len(sentence['alternative'].split()) >= self.min_tokens and len(sentence['alternative'].split()) <= self.max_tokens:
                    current_texts.append(sentence['alternative'])
                    current_lengths.append(len(sentence['alternative'].split()))
                    current_contexts.append(sentence['alternative_context'] if 'alternative_context' in sentence else None)
                    current_tags.append('<edit>')
                if self.max_sentences and len(current_texts) >= self.max_sentences:
                    exceeded_max_sentences = True
                    break
                    # return {'text': np.array(texts, dtype=object), 'length': np.array(lengths, dtype=object), 'context': np.array(contexts, dtype=object), 'tag': np.array(tags, dtype=object),
                    #             'number_in_par': np.arange(1,1+len(texts))}
            if exceeded_max_sentences:
                break
        
        # If exceede max sentences only if self.max_sentences is not None
        if (self.max_sentences and exceeded_max_sentences) or (not self.max_sentences):
            # If combined, combine the data
            if combined:
                texts = texts + current_texts
                lengths = lengths + current_lengths
                contexts = contexts + current_contexts
                tags = tags + current_tags
            else:
                texts.append(np.array(current_texts))
                lengths.append(np.array(current_lengths))
                contexts.append(np.array(current_contexts))
                tags.append(np.array(current_tags))
            
        return {'text': np.array(texts, dtype=object), 'length': np.array(lengths, dtype=object), 'context': np.array(contexts, dtype=object), 'tag': np.array(tags, dtype=object),
                    'number_in_par': np.arange(1,1+len(texts))}

