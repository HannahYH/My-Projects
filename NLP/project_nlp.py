## Import Libraries and Modules here...
import spacy
import math
import itertools
import collections

def allsubsets(S):
    sets = []
    for n in range(1, len(S) + 1):
        sets += [set(e) for e in itertools.combinations(S, n)]
    return sets

def is_entity_in(entity, Q):
    previous_index = 0
    pop_index = 0
    entity_is_in_Q = True
    words_of_entity = entity.split(' ')
    for word in words_of_entity:
        try:
            pop_index = Q.index(word, previous_index)
            Q.pop(pop_index)
            previous_index = pop_index
        except ValueError: # word cannot be found
            entity_is_in_Q = False
            break
    return entity_is_in_Q

class InvertedIndex:
    def __init__(self):
        ## You should use these variable to store the term frequencies for tokens and entities...
        self.tf_tokens = collections.defaultdict(dict)
        self.tf_entities = collections.defaultdict(dict)

        ## You should use these variable to store the inverse document frequencies for tokens and entities...
        self.idf_tokens = collections.defaultdict(dict)
        self.idf_entities = collections.defaultdict(dict)
        
        self.tfnorm_tokens = collections.defaultdict(dict)
        self.tfnorm_entities = collections.defaultdict(dict)

    ## Your implementation for indexing the documents...
    def index_documents(self, documents):
        ## Replace this line with your implementation...
        nlp = spacy.load("en_core_web_sm")
        doc_num = len(documents)
        
        # compute TF
        TFtoken = collections.defaultdict(dict)
        TFentity = collections.defaultdict(dict)
        for doc_key in documents.keys():
            single_entity = {}
            doc = nlp(documents[doc_key])
            for ent in doc.ents:
                if str(ent) in TFentity.keys():
                    if doc_key in TFentity[str(ent)].keys():
                        TFentity[str(ent)][doc_key] += 1
                    else:
                        TFentity[str(ent)][doc_key] = 1
                else:
                    TFentity.update({str(ent): {doc_key: 1}})
                
                words_in_one_entity = str(ent).split(' ')
                if len(words_in_one_entity) == 1:
                    if str(ent) in single_entity.keys():
                        single_entity[str(ent)] += 1
                    else:
                        single_entity[str(ent)] = 1
            for token in doc:
                if not token.is_stop and not token.is_punct:
                    if str(token) in TFtoken.keys():
                        if doc_key in TFtoken[str(token)].keys():
                            TFtoken[str(token)][doc_key] += 1
                        else:
                            TFtoken[str(token)][doc_key] = 1
                    else:
                        TFtoken.update({str(token): {doc_key: 1}})
            for single in single_entity:
                if single in TFtoken.keys():
                    TFtoken[single][doc_key] -= single_entity[single]
                    if TFtoken[single][doc_key] == 0 and len(TFtoken[single]) >= 1:
                        del TFtoken[single][doc_key]
                        if len(TFtoken[single]) == 0:
                            del TFtoken[single]
        self.tf_tokens = TFtoken
        self.tf_entities = TFentity
        
        # do normalization
        TFnorm_token = collections.defaultdict(dict)
        TFnorm_entity = collections.defaultdict(dict)
        for t in TFtoken:
            TFnorm_token[t] = dict()
            for d in TFtoken[t]:
                TFnorm_token[t][d] = 1.0 + math.log(1.0 + math.log(TFtoken[t][d]))
        for e in TFentity:
            TFnorm_entity[e] = dict()
            for d in TFentity[e]:
                TFnorm_entity[e][d] = 1.0 + math.log(TFentity[e][d])
        self.tfnorm_tokens = TFnorm_token
        self.tfnorm_entities = TFnorm_entity
        
        # compute IDF
        IDFtoken = collections.defaultdict(dict)
        IDFentity = collections.defaultdict(dict)
        for t in TFtoken:
            IDFtoken[t] = 1.0 + math.log(doc_num / (1.0 + len(TFtoken[t])))
        for e in TFentity:
            IDFentity[e] = 1.0 + math.log(doc_num / (1.0 + len(TFentity[e])))
        self.idf_tokens = IDFtoken
        self.idf_entities = IDFentity
            
    
    ## Your implementation to split the query to tokens and entities...
    def split_query(self, Q, DoE):
        token = Q.split(' ')
        query_splits = []
        ents = []
        # select entity presents in Q from DoE
        for item in DoE.keys():
            process_token = token[:]
            if is_entity_in(item, process_token):
                ents.append(item)
        # find all subsets
        subsets = allsubsets(ents)
        # find all valid entity sets and form the valid query split
        query_splits = {}
        q_id = 0
        for entity_set in subsets:
            entity_set = list(entity_set)
            # find all permutation for each subset
            permunated_entity_sets = itertools.permutations(entity_set)
            for each_entity_set in permunated_entity_sets:
                is_increasing = True
                process_token = token[:]
                for entity in each_entity_set:
                    # check if each entity in this subset present in Q
                    if not is_entity_in(entity, process_token):
                        is_increasing = False
                        break
                if is_increasing:
                    query_splits[q_id] = {'entities': entity_set, 'tokens': process_token}
                    q_id += 1
                    break
        query_splits[q_id] = {'entities': [], 'tokens': token}
        
        return query_splits



    ## Your implementation to return the max score among all the query splits...
    def max_score_query(self, query_splits, doc_id):
        ## Replace this line with your implementation...
        ## Output should be a tuple (max_score, {'tokens': [...], 'entities': [...]})
        max_score = 0
        final_split = {}
        for split in query_splits.values():
            s1 = 0
            s2 = 0
            for e in split['entities']:
                if e in self.tfnorm_entities.keys() and e in self.idf_entities.keys():
                    if doc_id in self.tfnorm_entities[e].keys():
                        s1 += self.tfnorm_entities[e][doc_id] * self.idf_entities[e]
            for t in split['tokens']:
                if t in self.tfnorm_tokens.keys() and t in self.idf_tokens.keys():
                    if doc_id in self.tfnorm_tokens[t].keys():
                        s2 += self.tfnorm_tokens[t][doc_id] * self.idf_tokens[t]
            if max_score < s1 + s2*0.4:
                max_score = s1 + s2*0.4
                final_split = split
        return (max_score, final_split)