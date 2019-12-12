import numpy as np
import xgboost as xgb
import spacy
import math
import collections

class InvertedIndex:
    def __init__(self):
        self.tf_tokens = collections.defaultdict(dict)
        self.idf_tokens = collections.defaultdict(dict)
        self.tfnorm_tokens = collections.defaultdict(dict)
        self.token_of_doc = collections.defaultdict(list)

    ## Your implementation for indexing the documents...
    def index_documents(self, documents):
        ## Replace this line with your implementation...
        nlp = spacy.load("en_core_web_sm")
        doc_num = len(documents)
        
        # compute TF
        TFtoken = collections.defaultdict(dict)
        for doc_key in documents.keys():
            doc = nlp(documents[doc_key]) #
            for token in doc:
                if not token.is_stop and not token.is_punct:
                    self.token_of_doc[doc_key].append(str(token))
                    if str(token) in TFtoken.keys():
                        if doc_key in TFtoken[str(token)].keys():
                            TFtoken[str(token)][doc_key] += 1
                        else:
                            TFtoken[str(token)][doc_key] = 1
                    else:
                        TFtoken.update({str(token): {doc_key: 1}})
        self.tf_tokens = TFtoken
        
        # do normalization
        TFnorm_token = collections.defaultdict(dict)
        for t in TFtoken:
            TFnorm_token[t] = dict()
            for d in TFtoken[t]:
                TFnorm_token[t][d] = 1.0 + math.log(1.0 + math.log(TFtoken[t][d]))
        self.tfnorm_tokens = TFnorm_token
        
        # compute IDF
        IDFtoken = collections.defaultdict(dict)
        for t in TFtoken:
            IDFtoken[t] = 1.0 + math.log(doc_num / (1.0 + len(TFtoken[t])))
        self.idf_tokens = IDFtoken
        
class InvertedIndex_Parsed_Entity:
    def __init__(self):
        self.tf_tokens = collections.defaultdict(dict)
        self.idf_tokens = collections.defaultdict(dict)
        self.tfnorm_tokens = collections.defaultdict(dict)

    ## Your implementation for indexing the documents...
    def index_documents(self, documents):
        ## Replace this line with your implementation...
        doc_num = len(documents)
        
        # compute TF
        TFtoken = collections.defaultdict(dict)
        for doc_key in documents.keys():
            doc = documents[doc_key] #
            for line in doc:
                token = line[1]
                if str(token) in TFtoken.keys():
                    if doc_key in TFtoken[str(token)].keys():
                        TFtoken[str(token)][doc_key] += 1
                    else:
                        TFtoken[str(token)][doc_key] = 1
                else:
                    TFtoken.update({str(token): {doc_key: 1}})
        self.tf_tokens = TFtoken
        
        # do normalization
        TFnorm_token = collections.defaultdict(dict)
        for t in TFtoken:
            TFnorm_token[t] = dict()
            for d in TFtoken[t]:
                TFnorm_token[t][d] = 1.0 + math.log(1.0 + math.log(TFtoken[t][d]))
        self.tfnorm_tokens = TFnorm_token
        
        # compute IDF
        IDFtoken = collections.defaultdict(dict)
        for t in TFtoken:
            IDFtoken[t] = 1.0 + math.log(doc_num / (1.0 + len(TFtoken[t])))
        self.idf_tokens = IDFtoken

def form_data(mentions, index, index_parsed_entity_pages, men_docs, parsed_entity_pages, label_in=None):
    groups = []
    candidate_ents = []
    feature_1 = []
    feature_2 = []
    feature_3 = []
    feature_4 = []
    labels = []
    spacy_nlp = spacy.load('en_core_web_sm')
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    for k, v in mentions.items():
        groups.append(len(v['candidate_entities']))
        men_docs_token = set(index.token_of_doc[v['doc_title']])
        for ent in v['candidate_entities']:
            tf_1 = 0.0
            tf_2 = 0.0
            # entities
            candidate_ents.append(ent)

            # feature 1: tf_idf score of each candidate entity in doc index
            for item in ent.split('_'):
                if item not in spacy_stopwords:
                    if v['doc_title'] in index.tf_tokens[item]:
                        tf_1 += index.tfnorm_tokens[item][v['doc_title']] * index.idf_tokens[item]
            feature_1.append(tf_1)
            
            # feature 2: tf_idf score of mention doc in parsed entity index
            for item in index.token_of_doc[v['doc_title']]:
                if item not in spacy_stopwords:
                    if ent in index_parsed_entity_pages.tf_tokens[item]:
                        tf_2 += index_parsed_entity_pages.tfnorm_tokens[item][ent] * index_parsed_entity_pages.idf_tokens[item]
            feature_2.append(tf_2)

            # feature_3: rate
            len_of_2_terms = len(ent) / len(v['mention'])
            feature_3.append(len_of_2_terms)
            
            # feature_4: common words length
            ent_token = set(index_parsed_entity_pages.tf_tokens[ent].keys())
            len_of_2_sets = len(men_docs_token & ent_token) / len(set(index.token_of_doc[v['doc_title']])) # normalise
            feature_4.append(len_of_2_sets)
            
            if label_in:
                # labels
                if label_in[k]['label'] == ent:
                    labels.append(1)
                else:
                    labels.append(0)

    groups = np.array(groups)
    candidate_ents = np.array([candidate_ents]).T 
    labels = np.array([labels]).T 
    feature_1 = np.array([feature_1]).T 
    feature_2 = np.array([feature_2]).T 
    feature_3 = np.array([feature_3]).T 
    feature_4 = np.array([feature_4]).T 
    data_features = np.concatenate((feature_1, feature_2, feature_3, feature_4), axis=1)
    return groups, data_features, labels
        
def disambiguate_mentions(train_mentions, train_labels, dev_mentions, men_docs, parsed_entity_pages):
    ## You can replace this line with your code...
    
    # data preprocess and index construction
    # mention document index
    index = InvertedIndex()
    index.index_documents(men_docs)
    # parsed entity index
    index_parsed_entity_pages = InvertedIndex_Parsed_Entity()
    index_parsed_entity_pages.index_documents(parsed_entity_pages)
    # generate features
    pred_group, pred_feature, pred_label = form_data(dev_mentions, index, index_parsed_entity_pages, men_docs, parsed_entity_pages)
    train_group, train_feature, train_label = form_data(train_mentions, index, index_parsed_entity_pages, men_docs, parsed_entity_pages, train_labels)
    
    #param = {'max_depth':8, 'eta':0.05, 'silent':1, 'objective':'rank:pairwise', 'min_child_weight':0.01, 'lambda':100}
    param = {'max_depth':8, 'eta':0.05, 'n_estimators':5000, 'objective':'rank:pairwise', 'min_child_weight':0.01, 'lambda':100}
    
    # train
    xgboost_train = xgb.DMatrix(data=train_feature, label=train_label)
    xgboost_train.set_group(train_group)
    classifier = xgb.train(param, xgboost_train, num_boost_round=4900)
    
    # test
    xgboost_test = xgb.DMatrix(data=pred_feature)
    xgboost_test.set_group(pred_group)
    preds = classifier.predict(xgboost_test)
    
    sum_ = 0
    result = {}
    for i in range(len(pred_group)):
        entity_idx = np.argmax(preds[sum_:sum_+pred_group[i]])
        entity = dev_mentions[i+1]['candidate_entities'][entity_idx]
        sum_ += pred_group[i]
        result[i+1] = entity

    return result