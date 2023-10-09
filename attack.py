import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertForMaskedLM
import re
from sentence_transformers import SentenceTransformer, util
import heapq
import math 
import csv

EPSILON = 1e-10  # 这是一个非常小的数，您可以根据需要调整它


# 选择最相似的同义词
def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count=10, threshold=0.):
    """
    Given a list of source words (their indices), a similarity matrix, and an index-to-word mapping,
    this function returns the top `ret_count` similar words for each source word, filtered by a given threshold.
    
    Parameters:
    - src_words: List of source word indices.
    - sim_mat: Similarity matrix of shape (vocab_size, vocab_size).
    - idx2word: A mapping from word index to actual word.
    - ret_count: Number of top similar words to return for each source word.
    - threshold: A similarity threshold to filter out words.
    
    Returns:
    - sim_words: A list of lists containing similar words for each source word.
    - sim_values: A list of lists containing similarity values for each word in sim_words.
    """
    
    # 对于每个src_word，找到其与其他所有单词的相似度排名（从高到低）
    sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
    
    sim_words, sim_values = [], []  # 初始化列表以保存结果

    # 遍历src_words的每个词
    for idx, src_word in enumerate(src_words):
        # 获取对应src_word的相似度值
        sim_value = sim_mat[src_word][sim_order[idx]]
        
        # 根据阈值筛选出大于等于threshold的相似度值
        mask = sim_value >= threshold
        
        # 使用mask获取单词和其相似度值
        sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
        
        # 将单词索引转换为实际的单词
        sim_word = [idx2word[id] for id in sim_word]
        
        # 保存结果
        sim_words.append(sim_word)
        sim_values.append(sim_value)

    return sim_words, sim_values  # 返回相似单词及其相似度值



# 选择最相似的同义词
def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count=10, threshold=0.):
    """
    Given a list of source words (their indices), a similarity matrix, and an index-to-word mapping,
    this function returns the top `ret_count` similar words for each source word, filtered by a given threshold.
    
    Parameters:
    - src_words: List of source word indices.
    - sim_mat: Similarity matrix of shape (vocab_size, vocab_size).
    - idx2word: A mapping from word index to actual word.
    - ret_count: Number of top similar words to return for each source word.
    - threshold: A similarity threshold to filter out words.
    
    Returns:
    - sim_words: A list of lists containing similar words for each source word.
    - sim_values: A list of lists containing similarity values for each word in sim_words.
    """
    
    # 对于每个src_word，找到其与其他所有单词的相似度排名（从高到低）
    sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
    
    sim_words, sim_values = [], []  # 初始化列表以保存结果

    # 遍历src_words的每个词
    for idx, src_word in enumerate(src_words):
        # 获取对应src_word的相似度值
        sim_value = sim_mat[src_word][sim_order[idx]]
        
        # 根据阈值筛选出大于等于threshold的相似度值
        mask = sim_value >= threshold
        
        # 使用mask获取单词和其相似度值
        sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
        
        # 将单词索引转换为实际的单词
        sim_word = [idx2word[id] for id in sim_word]
        
        # 保存结果
        sim_words.append(sim_word)
        sim_values.append(sim_value)

    return sim_words, sim_values  # 返回相似单词及其相似度值

# 计算影响最大的词
def influential_tokens(sentence, model, tokenizer, k=5):
    # 确保模型在评估模式
    model.eval()

    # 获取原始logit输出
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        original_logits = model(**inputs).logits

    # 为每个token计算logit差异
    token_ids = inputs["input_ids"][0].tolist()  # 获取token ids
    tokens = [tokenizer.decode([token_id]) for token_id in token_ids]  # 转换为tokens
    diffs = []

    for i, token_id in enumerate(token_ids):
        # 如果是[CLS], [SEP]或[PAD]，则跳过
        if token_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            continue

        # 将当前token替换为[MASK]
        masked_input_ids = inputs["input_ids"].clone()
        masked_input_ids[0][i] = tokenizer.mask_token_id

        # 获取mask后的logits
        with torch.no_grad():
            masked_logits = model(input_ids=masked_input_ids, attention_mask=inputs["attention_mask"]).logits

        # 计算logit差异
        diff = torch.abs(original_logits - masked_logits).sum().item()
        diffs.append((tokens[i], i, diff))  # 保存token, index和差异值

    # 根据差异大小排序tokens
    sorted_tokens = sorted(diffs, key=lambda x: x[2], reverse=True)

    # 返回前k个token及其索引
    return [(token_info[0], token_info[1]) for token_info in sorted_tokens[:k]]


# 正则表达式去掉特殊字符
def extract_words(sentence):
    words = re.findall(r'\b\w+\b', sentence)
    return ' '.join(words)

def get_token_from_encoded(sentence, index, tokenizer):
    token_ids = tokenizer.encode(sentence, add_special_tokens=True)
    token = tokenizer.decode([token_ids[index]])
    return token


def mask_sentence(sentence, mask_idx, tokenizer):
    """Inserts a [MASK] token at the specified index of the sentence."""
    tokens = tokenizer.tokenize(sentence)
    tokens[mask_idx] = '[MASK]'
    return tokenizer.convert_tokens_to_string(tokens)


def replace_token_at_index(sentence, index, replacement_word, tokenizer):
    token_ids = tokenizer.encode(sentence, add_special_tokens=False)
    replacement_ids = tokenizer.encode(replacement_word, add_special_tokens=False)
    
    if len(replacement_ids) != 1:
        # 将replacement_word设置为"[UNK]"标记
        replacement_word = "[UNK]"
        replacement_ids = tokenizer.encode(replacement_word, add_special_tokens=False)
    
    token_ids[index] = replacement_ids[0]
    return tokenizer.decode(token_ids)

def generate_nested_sentences(sentence, indices, replacement_words_nested_list, tokenizer):
    if len(indices) != len(replacement_words_nested_list):
        raise ValueError("Length of indices and replacement words nested list should be the same.")
    
    nested_sentences = []
    for idx, replacement_words_list in zip(indices, replacement_words_nested_list):
        modified_sentences_for_idx = []
        for replacement_word in replacement_words_list:
            modified_sentence = replace_token_at_index(sentence, idx, replacement_word, tokenizer)
            modified_sentences_for_idx.append(modified_sentence)
        nested_sentences.append(modified_sentences_for_idx)
    return nested_sentences

# 计算Logit输出
def compute_logits(sentence):
    inputs = adver_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = adver_model(**inputs)
    logits = outputs.logits
    return logits

# 计算概率输出
def compute_probabilities(sentence):
    logits = compute_logits(sentence)
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    return probabilities

# 计算M3概率矩阵
def compute_difference_matrix(original_sentence, nested_adversarial_sentences, nested_replacement_words):
    difference_matrix = []

    # 为原始句子计算softmax概率
    original_probs = compute_probabilities(original_sentence)
    #print(original_probs)
    original_pred_class = torch.argmax(original_probs, dim=1).item()
    original_pred_prob = original_probs[0, original_pred_class].item()

    for group_idx, group in enumerate(nested_adversarial_sentences):
        group_differences = {}

        for sentence_idx, sentence in enumerate(group):
            adversarial_probs = compute_probabilities(sentence)
            #print(adversarial_probs)

            adversarial_pred_class = torch.argmax(adversarial_probs, dim=1).item()
            adversarial_pred_prob = adversarial_probs[0, original_pred_class].item()
            #print(adversarial_pred_prob)

            # 计算概率差值
            difference = original_pred_prob - adversarial_pred_prob

            # 使用替换词作为key存储差值
            replacement_word = nested_replacement_words[group_idx][sentence_idx]
            group_differences[replacement_word] = difference
            #print(difference)
            # break
        difference_matrix.append(group_differences)
    return difference_matrix

def get_global_min(lst):
    """获取所有字典中的全局最小值"""
    global_min = float('inf')
    for d in lst:
        local_min = min(d.values())
        global_min = min(global_min, local_min)
    return global_min

def l1_normalize_dict(d, shift_value):
    """对偏移后的字典进行L1归一化"""
    shifted_values = [v + shift_value for v in d.values()]
    total = sum(shifted_values)
    
    if total == 0:
        return {k: 1.0/len(d) for k in d}

    normalized_dict = {k: shifted_val / total for k, shifted_val in zip(d.keys(), shifted_values)}
    
    # 替换为0的值
    for k, v in normalized_dict.items():
        if v == 0:
            normalized_dict[k] = EPSILON
            
    return normalized_dict

def l1_normalize_list_of_dicts(lst):
    """对整个列表的字典进行L1归一化"""
    global_min = get_global_min(lst)
    shift_value = -global_min if global_min < 0 else 0
    return [l1_normalize_dict(d, shift_value) for d in lst]

def replace_tokens_in_sentence(sentence, indices, replacement_tokens, tokenizer):
    tokens = tokenizer.tokenize(sentence)
    for i, index in enumerate(indices):
        tokens[index-1] = replacement_tokens[i]
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens))

def predict_sentiment(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class


def k_best_viterbi(probs, k=1):
    paths = [(score, [token]) for token, score in probs[0].items()]
    for i in range(1, len(probs)):
        new_paths = []
        for prev_score, prev_path in paths:
            for next_token, next_score in probs[i].items():
                new_score = prev_score + next_score
                heapq.heappush(new_paths, (new_score, prev_path + [next_token]))
                if len(new_paths) > k:
                    heapq.heappop(new_paths)
        paths = new_paths
    return sorted(paths, key=lambda x: x[0], reverse=True)




# ------------------load something----------------------------------

# 加载同义词集
cos_sim = np.load('cos_sim_counter_fitting.npy')


# 加载窃取模型
adver_path = "/home/ubuntu/zhc/work/adversarial_output"
adver_tokenizer = BertTokenizer.from_pretrained(adver_path)
adver_model = BertForSequenceClassification.from_pretrained(adver_path)


# load black-box model
black_box_path = "/home/ubuntu/zhc/work/output"
black_tokenizer = BertTokenizer.from_pretrained(black_box_path)
black_model = BertForSequenceClassification.from_pretrained(black_box_path)
black_model.eval()
black_model.to('cuda')  # if you are using GPU

# load mlm_model
model_mlm = "bert-base-uncased"
mlm_model = BertForMaskedLM.from_pretrained(model_mlm)
mlm_tokenizer = BertTokenizer.from_pretrained(model_mlm)
mlm_model.eval()

# laod similar  
similar_model = SentenceTransformer('bert-base-nli-mean-tokens')
# load clean
input_file_path = "/home/ubuntu/zhc/work/data/clean/clean.tsv"
output_file_path = "/home/ubuntu/zhc/work/data/clean/successful_attacks.tsv"

successful_attacks = 0
total_attacks = 0
k = 2 # 攻击词数    


# 同义词表的idx2word and it's reverse
idx2word = {}
word2idx = {}

print("Building vocab...")
with open('./counter-fitted-vectors.txt', 'r') as ifile:
    for line in ifile:
        word = line.split()[0]
        if word not in idx2word:
            idx2word[len(idx2word)] = word
            word2idx[word] = len(idx2word) - 1
        
with open(input_file_path, 'r', newline='', encoding='utf-8') as input_file, \
     open(output_file_path, 'w', newline='', encoding='utf-8') as output_file:

    tsv_reader = csv.reader(input_file, delimiter='\t')
    tsv_writer = csv.writer(output_file, delimiter='\t')

    # Copy headers to the output file
    headers = next(tsv_reader)
    tsv_writer.writerow(headers)

    for row in tsv_reader:
        sentence = row[0]  # 假设每行的第一个字段是要攻击的句子
         # 攻击的句子
        print(sentence)
        sentence = extract_words(sentence)

        
        # 使用窃取模型计算top_k影响的词
        top_k_tokens_and_indices = influential_tokens(sentence, adver_model, adver_tokenizer, k)


        # 得到扰动词下标和词
        perturb_idxes = []
        words_perturb = []
        for i in range(k):
            perturb_idxes.append(top_k_tokens_and_indices[i][1])
            words_perturb.append(top_k_tokens_and_indices[i][0])
            

        # 同义词典的位置   
        words_perturb_idx = [word2idx[word] for word in words_perturb if word in word2idx]
        # 同义词index及其名称
        words_perturb = [(idx, idx2word[idx]) for idx in words_perturb_idx]
        # 得到同义词
        synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, 200, 0.1)


        #-----------------M1--------------------------
        word_mask_probability_list = []

        for index, idx in enumerate(perturb_idxes):    
            msk_sentence = mask_sentence(sentence, mask_idx=idx-1, tokenizer=mlm_tokenizer)
            
            inputs = mlm_tokenizer(msk_sentence, return_tensors="pt")
            mask_idx = torch.where(inputs["input_ids"][0] == mlm_tokenizer.mask_token_id)[0].item()
            # 3. 使用BERT预测[MASK]位置的概率分布
            with torch.no_grad():
                outputs = mlm_model(**inputs)
                predictions = outputs.logits[0, mask_idx].softmax(dim=0)

            word_mask_probability = {}
            
            for word in synonym_words[index]:
                word_id = mlm_tokenizer.convert_tokens_to_ids(word)
                word_probability = predictions[word_id].item()
                word_mask_probability[word] = word_probability

            
            total_value = sum(word_mask_probability.values())
            normalized_data = {k: v / total_value for k, v in word_mask_probability.items()}

            word_mask_probability_list.append(normalized_data)   
            
            
        #----------------------------M2---------------------------------

        indices = perturb_idxes
        replacement_words_nested_list = synonym_words

        nested_sentences = generate_nested_sentences(sentence, indices, replacement_words_nested_list, mlm_tokenizer)

        word_similarity_list = []
        # 对单一句子进行编码
        query_embedding = similar_model.encode(sentences=sentence, convert_to_tensor=True)

        for i in range(len(nested_sentences)):
            # 对多个句子进行编码
            sentence_embeddings = similar_model.encode(nested_sentences[i], convert_to_tensor=True)
                
            # 计算相似度
            similarity_scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)
            similarity_scores = similarity_scores.to('cpu').numpy().reshape(-1)
            
            word_similarity_probability = {}
            for idx, word in enumerate(synonym_words[i]):    # break
                word_similarity_probability[word] = similarity_scores[idx]
            total_value = sum(word_similarity_probability.values())
            normalized_data = {k: v / total_value for k, v in word_similarity_probability.items()}

            word_similarity_list.append(normalized_data)   
            
            
        #------------------------------M3----------------------------------------------

        # 假设你已经有一个嵌套的对抗句子列表
        original_sentence = sentence
        nested_adversarial_sentences = nested_sentences

        # 假设你有一个替换词列表与上面的对抗句子列表相对应
        nested_replacement_words = synonym_words
        difference_matrix = compute_difference_matrix(original_sentence, nested_adversarial_sentences, nested_replacement_words)

        normalized_lst = l1_normalize_list_of_dicts(difference_matrix)
        normalized_lst


        # M1->word_similarity_list M2->word_mask_probability_list M3->normalized_lst



        combined_probs = []

        # 添加dict3到zip中
        for dict1, dict2, dict3 in zip(word_similarity_list, word_mask_probability_list, normalized_lst): 
            combined_dict = {}
            for token in dict1:
                # 对三个字典中的对应值求和
                combined_dict[token] = math.log(dict1[token]) + math.log(dict2[token]) + 2*math.log(dict3[token])
            combined_probs.append(combined_dict)


        #-----------------------attack---------------------------
        original_sentiment = predict_sentiment(sentence, black_model, black_tokenizer)

        max_attempts = 1000
        success = False
        attempted_paths = []

        for attempt in range(1, max_attempts+1):
            all_paths = k_best_viterbi(combined_probs, k=max_attempts)
            best_path_not_attempted = next((path for path in all_paths if path[1] not in attempted_paths), None)

            if not best_path_not_attempted:
                break

            score, path = best_path_not_attempted
            attempted_paths.append(path)
            modified_sentence = replace_tokens_in_sentence(sentence, indices, path, black_tokenizer)
            modified_sentiment = predict_sentiment(modified_sentence, black_model, black_tokenizer)

            if modified_sentiment != original_sentiment:
                print(f"Attack successful with sentence: {modified_sentence}")
                success = True
                break
            
            print(attempt)
            # 降低已经尝试路径的得分，以便在下次迭代中不再选择它们
            for idx, token in enumerate(path):
                combined_probs[idx][token] -= 1e6  # 使其得分显著降低
                
        if success:
            successful_attacks += 1
            row[0] = modified_sentence
            tsv_writer.writerow(row)
            
        if not success:
            print("Attack failed after trying all paths.")
            