from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import torch
from sklearn.cluster import KMeans
import json
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#file_path = 'messages.json'

# with open(file_path, 'r') as file:
#     data = json.load(file)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


def remove_html_tags(text):
    clean = re.compile('<.*?>|&nbsp;')
    return re.sub(clean, '', text).replace('\n', ' ')


def load_chats(): 
    chats = []
    temp_data = []
    with open('messages.json', 'r') as file:
        temp_data = json.load(file)
    sample_data = temp_data[:20]
    chats = [remove_html_tags(d['message']) for d in sample_data]
    return chats


def load_questions(): 
    questions_data = []
    with open('questions.json', 'r') as file:
        questions_data = json.load(file)
    return questions_data



def generate_bert_embeddings(text):
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')
    with torch.no_grad():
        outputs = bert_model(input_ids)
        embeddings = outputs[0][:, 0, :].squeeze(0).numpy() 
    return embeddings



def generate_sentence_transformers_embeddings(sentences):
    embeddings = sentence_model.encode(sentences)
    return embeddings



def compare_similarity(compare_embedding, sample_embeddings): 
    similarities = cosine_similarity(compare_embedding, sample_embeddings)[0]
    return similarities



def print_similarity(questions, similarities, frequency_of_questions):
    sorted_messages = [(questions[i], similarities[i], i) for i in range(len(questions))]
    sorted_messages.sort(key=lambda x: x[1], reverse=True)
    ind_of_top_matched_question = sorted_messages[0][2]
    frequency_of_questions[ind_of_top_matched_question] += 1
    top_5_sorted_messages = sorted_messages[:5]
    for message, similarity, ind in top_5_sorted_messages:
        print(f"Question: {message}") 
        print(f"Similarity: {int(similarity * 100)}")
        print(f"Frequency for this Question: {frequency_of_questions[ind]}")
        print()



def ask_questions():
    questions_data = load_questions()
    frequency_of_questions_bert = np.zeros(len(questions_data), dtype=int)
    frequency_of_questions_sentence = np.zeros(len(questions_data), dtype=int)
    while True:
        try: 
            user_input = int(input("Enter 1 to Use Bert model and 2 for sentence transformer: "))
            if user_input == 1: 
                sample_question = input('Enter question: ')
                question_embeddings = [generate_bert_embeddings(question) for question in questions_data]
                sample_question_embedding = generate_bert_embeddings(sample_question)
                similarities = compare_similarity([sample_question_embedding], question_embeddings)
                print_similarity(questions_data, similarities, frequency_of_questions_bert)
            elif user_input == 2:
                sample_question = input('Enter question: ')
                question_embeddings = generate_sentence_transformers_embeddings(questions_data)
                sample_question_embedding = generate_sentence_transformers_embeddings(sample_question)
                similarities = compare_similarity([sample_question_embedding], question_embeddings)
                print_similarity(questions_data, similarities, frequency_of_questions_sentence)
            else:
                print('Invalid input, exiting')
                return
        except ValueError: 
            print('Invalid input, exiting')
            return


def run_chats():
    questions_data = load_questions()
    frequency_of_questions_bert = np.zeros(len(questions_data), dtype=int)
    frequency_of_questions_sentence = np.zeros(len(questions_data), dtype=int)
    chats = load_chats()
    user_input = int(input("Enter 1 to Use Bert model and 2 for sentence transformer: "))
    try: 
        if user_input == 1: 
            question_embeddings = [generate_bert_embeddings(question) for question in questions_data]
            for chat in chats: 
                sample_question = chat
                print('Question')
                print(sample_question)
                print()
                sample_question_embedding = generate_bert_embeddings(sample_question)
                similarities = compare_similarity([sample_question_embedding], question_embeddings)
                print_similarity(questions_data, similarities, frequency_of_questions_bert)
        elif user_input == 2:
            question_embeddings = generate_sentence_transformers_embeddings(questions_data)
            for chat in chats: 
                sample_question = chat
                print('Message')
                print(sample_question)
                print()
                sample_question_embedding = generate_sentence_transformers_embeddings(sample_question)
                similarities = compare_similarity([sample_question_embedding], question_embeddings)
                print_similarity(questions_data, similarities, frequency_of_questions_sentence)
        else:
            print('Invalid input, exiting')
            return
    except ValueError: 
        print('Invalid input, exiting')
        return


#ask_questions()
    
run_chats()


# question_embeddings = [generate_bert_embeddings(question) for question in questions_data]
# sample_question = 'Consent forms for ipr?'
# sample_question_embedding = generate_bert_embeddings(sample_question)
# sample_question_embedding = generate_sentence_transformers_embeddings(sample_question)
# similarities = compare_similarity([sample_question_embedding], question_embeddings)
# print_similarity(questions_data, similarities, frequency_of_questions)

# messages = [remove_html_tags(d.get('message')) for d in data]
# messages_length = len(messages)


# num_clusters = 100  
# kmeans = KMeans(n_clusters=num_clusters)
# clusters = kmeans.fit_predict(embeddings)
# print('clusters')
# print(clusters)
# print()

# grouped_messages = [[] for _ in range(num_clusters)]
# for x in range(0, messages_length):
#     cluster_ind = clusters[x]
#     grouped_messages[cluster_ind].append(messages[x])

#print('grouped_messages')
# for item in grouped_messages:
#     #print('grouped')
#     for message in item:
#         # print(message)
#         # print()
#     # print()


# grouped_file_path = 'grouped_messages.json'
# with open(grouped_file_path, "w") as json_file:
#     json.dump(grouped_messages, json_file, indent=4)