import csv
import math
import nltk
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
stop_words = stopwords.words('english')
clf = SGDClassifier(loss='log', random_state=1)


vect = HashingVectorizer(decode_error='ignore', n_features=2**21, preprocessor=None, tokenizer=None)
tfidf_vec = TfidfVectorizer()
wnl = WordNetLemmatizer()

def get_task_dataset(infile):
    training_dict = {}
    csv.field_size_limit(500 * 1024 * 1024)
    print("\nReading and getting task training set from {}".format(infile))
    with open(infile ,'r', encoding='UTF-8') as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',')
        line_count=0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                UID = row[1]
                if UID in training_dict:
                    training_dict[UID].append(row[0])
                else:
                    training_dict[UID] = [row[0]]
    return training_dict #{user_id:[post_ids]}

def get_shared_task_posts(infile):
    post_dict = {}
    csv.field_size_limit(500 * 1024 * 1024)
    print("\nReading and getting post set from {}".format(infile))
    with open(infile,'r', encoding='UTF-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        line_count=0
        for row in csv_reader:
            if line_count==0:
                line_count+=1
            else:
                post_dict[row[0]]=[row[1],row[2],row[3],row[4],row[5]]
    return post_dict #{post_id:[user_id,timestamp,subreddit,post_title,post_body]}


def find_all_posts_for_user(task_dict,shared_dict):
    print('\nfinding all post in the task')
    all_posts = {}
    for key, value in task_dict.items():
        if int(key)>0:
            all_posts[key] = []
            for postID in value:
                all_posts[key].append((shared_dict[postID])[4])
    return all_posts #{userID:[posts]}

def cleaning_and_tokenlize_post_test(posts_body_text):
    tokens_list=[]
    for post in posts_body_text:
        words = word_tokenize(post)
        #keep only alphabet(also remove 's maybe 't)
        tokens = [word for word in words if word.isalpha()]
        #conver to lowercase
        tokens = [w.lower() for w in tokens]
        #filter out stop words
        tokens = [w for w in tokens if not w in stop_words]
        #Lemmatization
        tokens = [wnl.lemmatize(w) for w in tokens]
        #=======================discuss here in report=================================#
        #Stemming to reduce each word to its root or base
        #tokens = [porter.stem(word) for word in tokens]
        for token in tokens:
            tokens_list.append(token)
    listToStr = ','.join(map(str,tokens_list))
    return listToStr

def find_user_token_list(userposts):
    user_tokens = {}
    for key, value in userposts.items():
        if int(key)>0:
            user_tokens[key]=cleaning_and_tokenlize_post_test(value)
    return user_tokens #{userID:[tokens]}
        

def get_userid_label(infile):
    userid_label_dict = {}
    csv.field_size_limit(500 * 1024 * 1024)
    print("\nReading and getting user ID and label {}".format(infile))
    with open(infile,'r', encoding='UTF-8') as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',')
        line_count=0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                userid_label_dict[row[0]]=row[1]
    return userid_label_dict #{userID: Label}

def set_label_as_post(posts, labels):
    post = []
    for item1, item2 in zip(posts, labels):
        post.append(item1 + "," + str(item2))
    return post

def get_all(label_post_list):
    all_posts = [",".join(x for x in label_post_list)][0].split(",")
    return all_posts

def get_d_posts(posts_list):
    d_posts = [i for i in posts_list if "1" in i]
    d_posts = [",".join(i for i in d_posts)][0].split(",")
    return d_posts

# get P(w,c), i.e,
# the probability of word w occurs in class c
# how many times w occurs in c / how many times w occurs
def get_word_label_prob(word_in_label_dict, word_dict):
    prob = {}
    for k in word_in_label_dict.keys():
        prob[k] = word_in_label_dict[k]/word_dict[k]
    prob = dict(sorted(prob.items(), key=lambda item: item[1], reverse=True))
    return prob

# get P(w)  
# the number of word w / total number of words  
def get_word_prob(word_dict):
    prob = {}
    total = sum(word_dict.values())
    for k,v in word_dict.items():
        prob[k] = v/total
    prob = dict(sorted(prob.items(), key=lambda item: item[1], reverse=True))
    return prob

# get P(c)
# the number of label c / total number of labels
def get_label_prob(label_list, label):
    label_dict = Counter(label_list)
    total = sum(label_dict.values())
    prob = label_dict[label]/total
    return prob

# get PMI = P(w,c)/(P(w)*P(c))
def get_PMI_score(wc_prob, w_prob, c_prob):
    score = {}
    for k in wc_prob.keys():
        score[k] = math.log(wc_prob[k]/(w_prob[k]*c_prob))
    score = dict(sorted(score.items(), key=lambda item: item[1], reverse=True))
    return score

def user_pmi_tokens(posts_list, pmi_tokens_list):
    temp_list = []
    final_tokens = []
    for i in range(len(posts_list)):
        list_element = list(set(posts_list[i].split(",")).intersection(pmi_tokens_list))
        temp_list.append(list_element)
    for item in temp_list:
        final_tokens.append(",".join(item))
    return final_tokens


def get_userID_label_pair_in_task(crowd_userLabel, task_set):
    task_userID_label = {}
    for key in task_set:
        if int(key)>0 and key in crowd_userLabel.keys():
            task_userID_label[key] = crowd_userLabel[key]
    return task_userID_label #{userID:Label}


# set label level a as 0. b, c, d as 1. (changed)
# According to the project description,
# taskA would be a binary classification of 
# severe-risk (d) versus the lower-risk categories (a-c).
# Therefore, labels are changed to 
# a-c as 0 and d as 1.

def binary_label(userid_label):
    user_dict={}
    for key in userid_label:
        if int(key)>0:
            if userid_label[key] == 'd':
                user_dict[key] = 1
            else:
                user_dict[key] = 0
    return user_dict #{userID:binary label}

def training_tokens_and_label(userID_tokens, userID_binary_label):
    training_tokens = []
    label = []
    for key in userID_tokens:
        if int(key)>0 and key in userID_binary_label.keys():
            training_tokens.append(userID_tokens[key])
            label.append(userID_binary_label[key])
    return training_tokens, label



input_task_A_train_file = './crowd/train/task_A_train.posts.csv'
input_task_A_test_file = './crowd/test/task_A_test.posts.csv'

input_task_B_train_file = './crowd/train/task_B_train.posts.csv'
input_task_B_test_file = './crowd/test/task_B_test.posts.csv'

input_shared_posts_file = './crowd/train/shared_task_posts.csv'
input_shared_posts_test_file = './crowd/test/shared_task_posts_test.csv'

input_crowd_training_label = './crowd/train/crowd_train.csv'
input_crowd_testing_label = './crowd/test/crowd_test.csv'

input_expert_posts_file = "./expert/expert_posts.csv"
input_expert_label = "./expert/expert.csv"

def main():
    Shared_pool_train = get_shared_task_posts(input_shared_posts_file) #{post_id:[user_id,timestamp,subreddit,post_title,post_body]}
    Shared_pool_test = get_shared_task_posts(input_shared_posts_test_file)
    crowd_userID_label = get_userid_label(input_crowd_training_label) #{userID:Label}
    crowd_userID_label_test = get_userid_label(input_crowd_testing_label)

    #Get Task A training dataset
    taskA_train = get_task_dataset(input_task_A_train_file) #{user_id:[post_ids]}
    taskA_test = get_task_dataset(input_task_A_test_file)
    
    taskA_user_posts_train = find_all_posts_for_user(taskA_train,Shared_pool_train) #{userID:[posts]}
    taskA_user_posts_test = find_all_posts_for_user(taskA_test,Shared_pool_test)
    
    taskA_user_label = get_userID_label_pair_in_task(crowd_userID_label,taskA_train) #{userID:Label}
    taskA_user_label_test = get_userID_label_pair_in_task(crowd_userID_label_test,taskA_test)
    
    taskA_user_binary_Label = binary_label(taskA_user_label) #{userID:binary label}
    taskA_user_binary_Label_test = binary_label(taskA_user_label_test)
    
    taskA_user_tokens = find_user_token_list(taskA_user_posts_train) #{userID:[tokens]}
    taskA_user_tokens_test = find_user_token_list(taskA_user_posts_test)

    TaskA_train, TaskA_train_label = training_tokens_and_label(taskA_user_tokens,taskA_user_binary_Label)
    TaskA_test, TaskA_test_label = training_tokens_and_label(taskA_user_tokens_test,taskA_user_binary_Label_test)

    #Get Task B training dataset
    taskB_train = get_task_dataset(input_task_B_train_file) #{user_id:[post_ids]}
    taskB_test = get_task_dataset(input_task_B_test_file)
    taskB_user_posts_train = find_all_posts_for_user(taskB_train,Shared_pool_train) #{userID:[posts]}
    taskB_user_posts_test = find_all_posts_for_user(taskB_test,Shared_pool_test)
    taskB_user_label = get_userID_label_pair_in_task(crowd_userID_label,taskB_train) #{userID:Label}
    taskB_user_label_test = get_userID_label_pair_in_task(crowd_userID_label_test,taskB_test)
    taskB_user_binary_Label = binary_label(taskB_user_label) #{userID:binary label}
    taskB_user_binary_Label_test = binary_label(taskB_user_label_test)
    taskB_user_tokens = find_user_token_list(taskB_user_posts_train) #{userID:[tokens]}
    taskB_user_tokens_test = find_user_token_list(taskB_user_posts_test)

    TaskB_train, TaskB_train_label = training_tokens_and_label(taskB_user_tokens,taskB_user_binary_Label)
    TaskB_test, TaskB_test_label = training_tokens_and_label(taskB_user_tokens_test,taskB_user_binary_Label_test)

    #Expert dataset
    Expert_Pool = get_shared_task_posts(input_expert_posts_file)
    expert_user_postID = get_task_dataset(input_expert_posts_file)
    expert_user_posts = find_all_posts_for_user(expert_user_postID,Expert_Pool)
    expert_UID_Label = get_userid_label(input_expert_label)
    expert_UID_Binary_Label = binary_label(expert_UID_Label)
    expert_UID_tokenlist = find_user_token_list(expert_user_posts)

    # Calculate PMI scores between words and labels
    #P(c)
    num_label_test = Counter(TaskA_test_label)
    label_prob_test = get_label_prob(num_label_test, 1) # the probability of label 1 occurs
    num_label_train = Counter(TaskA_train_label)
    label_prob_train = get_label_prob(num_label_train, 1)

    #P(w)
    word_dict_test = Counter(get_all(set_label_as_post(TaskA_test, TaskA_test_label)))
    word_prob_test = get_word_prob(word_dict_test) 
    word_dict_train = Counter(get_all(set_label_as_post(TaskA_train, TaskA_train_label)))
    word_prob_train = get_word_prob(word_dict_train) 
    #P(w,d)
    d_word_dict_test = Counter(get_d_posts(set_label_as_post(TaskA_test, TaskA_test_label)))
    d_word_dict_train = Counter(get_d_posts(set_label_as_post(TaskA_train, TaskA_train_label)))
    
    word_prob_test = get_word_label_prob(d_word_dict_test, word_dict_test)# the probability of word w cooccurs with label 1
    word_prob_train = get_word_label_prob(d_word_dict_train, word_dict_train)

    # PMI scores for every word w
    PMI_test = get_PMI_score(word_prob_test, word_prob_test, label_prob_test)
    PMI_train = get_PMI_score(word_prob_train, word_prob_train, label_prob_train)
    
    # Select tokens that only appeared in the PMI score dictionary from each user's posts 
    post_PMI_test = user_pmi_tokens(TaskA_test, list(PMI_test.keys()))
    post_PMI_train = user_pmi_tokens(TaskA_train, list(PMI_train.keys()))
    
    #Task A, accuracy
    classes = np.array([0, 1])
    Hashing_vec_train = vect.transform(TaskA_train)
    Hashing_vec_test = vect.transform(TaskA_test)

    #cross validation for Task A
    loss_functions = ['hinge','log','modified_huber']
    validation_fractions = [0.1,0.3,0.5]
    cv_scores = []
    for loss in loss_functions:
        for vf in validation_fractions:
            clf = SGDClassifier(loss=loss, validation_fraction=vf,random_state = 1)
            cv_score = cross_val_score(clf,Hashing_vec_train,TaskA_train_label,cv=10,scoring='accuracy')
            cv_scores.append(cv_score.mean())
    #print(cv_scores)
    clf = SGDClassifier(loss='log', validation_fraction = 0.1, random_state=1)
    cv_score = cross_val_score(clf,Hashing_vec_train,TaskA_train_label,cv=10,scoring='accuracy')
    print('The 10-fold cross validation score: %.3f' %cv_score.mean())

    clf.partial_fit(Hashing_vec_train, TaskA_train_label,classes=classes)
    baseline_pred = clf.predict(Hashing_vec_test)
    print('Task A Baseline Accuracy: %.3f' % clf.score(Hashing_vec_test, TaskA_test_label))
    print(classification_report(TaskA_test_label, baseline_pred))
    print("Baseline Confusion Matrix: \n")
    print(confusion_matrix(TaskA_test_label, baseline_pred))

    label = ["No Risk", "Severe  Risk"]
    plot_confusion_matrix(clf, Hashing_vec_test, TaskA_test_label, display_labels = label)
    plt.title('Confusion matrix of the Baseline SGD')
    plt.show()

    #Evaluation with expert dataset
    expert_dataset, expert_label = training_tokens_and_label(expert_UID_tokenlist,expert_UID_Binary_Label)
    Hashing_vec_expert = vect.transform(expert_dataset)
    expert_pred = clf.predict(Hashing_vec_expert)
    cv_scores = cross_val_score(clf,Hashing_vec_expert,expert_label,cv=10,scoring='accuracy')
    print('The 10-fold cross validation mean score: %.3f' %cv_scores.mean())
    print('Task A Expert Evaluation Accuracy: %.3f' %  clf.score(Hashing_vec_expert, expert_label))
    print(classification_report(expert_label, expert_pred))
    print("Expert Evaluation Confusion Matrix: \n")
    print(confusion_matrix(expert_label, expert_pred))
    plot_confusion_matrix(clf, Hashing_vec_expert, expert_label, display_labels = label)
    plt.title('Confusion matrix of the Baseline SGD Expert Evaluation')
    plt.show()

    #Task B, accuracy
    TaskB_train = vect.transform(TaskB_train)
    TaskB_test = vect.transform(TaskB_test)
    classes = np.array([0, 1])

    #cross validation for Task B
    loss_functions = ['hinge','log','modified_huber']
    validation_fractions = [0.1,0.3,0.5]
    cv_scores = []
    for loss in loss_functions:
        for vf in validation_fractions:
            clf = SGDClassifier(loss=loss, validation_fraction=vf,random_state = 1)
            cv_score = cross_val_score(clf,TaskB_train,TaskB_train_label,cv=10,scoring='accuracy')
            cv_scores.append(cv_score.mean())
    #print(cv_scores)

    clf = SGDClassifier(loss='hinge', validation_fraction = 0.1, random_state=1)
    cv_score = cross_val_score(clf,TaskB_train,TaskB_train_label,cv=10,scoring='accuracy')
    print('The 10-fold cross validation score: %.3f' %cv_score.mean())

    clf.partial_fit(TaskB_train, TaskB_train_label,classes=classes)
    baseline_pred_TaskB = clf.predict(TaskB_test)
    print('Task B Accuracy: %.3f' % clf.score(TaskB_test, TaskB_test_label))
    print(classification_report(TaskB_test_label, baseline_pred_TaskB))
    label = ["No Risk", "Severe  Risk"]
    plot_confusion_matrix(clf, TaskB_test, TaskB_test_label, display_labels = label)
    plt.title('Confusion matrix of the Baseline SGD for Task B')
    plt.show()

    #Evaluation with expert dataset for Task B
    expert_dataset, expert_label = training_tokens_and_label(expert_UID_tokenlist,expert_UID_Binary_Label)
    Hashing_vec_expert = vect.transform(expert_dataset)
    expert_pred = clf.predict(Hashing_vec_expert)
    print('Task B Expert Evaluation Accuracy: %.3f' %  clf.score(Hashing_vec_expert, expert_label))
    print(classification_report(expert_label, expert_pred))
    print("Expert Evaluation Confusion Matrix for Task B: \n")
    print(confusion_matrix(expert_label, expert_pred))
    plot_confusion_matrix(clf, Hashing_vec_expert, expert_label, display_labels = label)
    plt.title('Confusion matrix of the Baseline SGD Expert Evaluation for Task B')
    plt.show()
    
    #TF-IDF
    Tfidf_train = tfidf_vec.fit_transform(TaskA_train)    
    Tfidf_test = tfidf_vec.transform(TaskA_test)

    #check TDIDF SOCRE
    #TDIDF_socre = sorted(zip(tfidf_vec.get_feature_names(),np.asarray(Tfidf_train.sum(axis=0)).ravel()),key=lambda x:x[1],reverse=True)
    #print(TDIDF_socre[:20]) 


    #cross validation for Task A TF-IDF
    loss_functions = ['hinge','log','modified_huber']
    validation_fractions = [0.1,0.3,0.5]
    cv_scores = []
    for loss in loss_functions:
        for vf in validation_fractions:
            clf = SGDClassifier(loss=loss, validation_fraction=vf,random_state = 1)
            cv_score = cross_val_score(clf,TaskB_train,TaskB_train_label,cv=10,scoring='accuracy')
            cv_scores.append(cv_score.mean())
    #print(cv_scores)

    clf = SGDClassifier(loss='hinge', validation_fraction = 0.1, random_state=1)
    cv_score = cross_val_score(clf,TaskB_train,TaskB_train_label,cv=10,scoring='accuracy')
    print('The 10-fold cross validation score: %.3f' %cv_score.mean())

    clf.fit(Tfidf_train, TaskA_train_label)
    Tfidf_pred = clf.predict(Tfidf_test)
    print('Task A TF-IDF SGD Classifier Accuracy: %.3f' %  clf.score(Tfidf_test, TaskA_test_label))
    print(classification_report(TaskA_test_label, Tfidf_pred))
    print("TFIDF Confusion Matrix: \n")
    print(confusion_matrix(TaskA_test_label, Tfidf_pred))
    plot_confusion_matrix(clf, Tfidf_test, TaskA_test_label, display_labels = label)
    plt.title('Confusion matrix of the TF-IDF SGD')
    plt.show()
    #Evaluation with expert dataset on TF-IDF based features
    Tfidf_expert_test = tfidf_vec.transform(expert_dataset)
    Tfidf_expert_pred = clf.predict(Tfidf_expert_test)

    print('Task A Expert Evaluation Accuracy on TF-IDF based features: %.3f' %  clf.score(Tfidf_expert_test, expert_label))
    print(classification_report(expert_label, expert_pred))
    print("Expert Evaluation Confusion Matrix: \n")
    print(confusion_matrix(expert_label, Tfidf_expert_pred))
    plot_confusion_matrix(clf, Tfidf_expert_test, expert_label, display_labels = label)
    
    plt.title('Confusion matrix of the TF-IDF SGD Expert Evaluation')
    plt.show()

    #PMI
    TaskA_PMI_train = vect.transform(post_PMI_train)
    TaskA_PMI_test = vect.transform(post_PMI_test)

    #cross validation for Task A TF-IDF
    loss_functions = ['hinge','log','modified_huber']
    validation_fractions = [0.1,0.3,0.5]
    cv_scores = []
    for loss in loss_functions:
        for vf in validation_fractions:
            clf = SGDClassifier(loss=loss, validation_fraction=vf,random_state = 1)
            cv_score = cross_val_score(clf,TaskA_PMI_train,TaskA_train_label,cv=10,scoring='accuracy')
            cv_scores.append(cv_score.mean())
    #print(cv_scores)

    clf = SGDClassifier(loss='log', validation_fraction = 0.3, random_state=1)
    cv_score = cross_val_score(clf,TaskB_train,TaskB_train_label,cv=10,scoring='accuracy')
    print('The 10-fold cross validation score: %.3f' %cv_score.mean())
    
    clf.fit(TaskA_PMI_train, TaskA_train_label)
    TaskA_PMI_pred = clf.predict(TaskA_PMI_test)
    print('Task A PMI SGD Classifier Accuracy: %.3f' %  clf.score(TaskA_PMI_test, TaskA_test_label))
    print(classification_report(TaskA_test_label, TaskA_PMI_pred))
    print("PMI Confusion Matrix: \n")
    print(confusion_matrix(TaskA_test_label, TaskA_PMI_pred))
    plot_confusion_matrix(clf, TaskA_PMI_test, TaskA_test_label, display_labels = label)
    plt.title('Confusion matrix of the PMI SGD')
    plt.show()
    
    #Evaluation with expert dataset on PMI based features
    expert_dataset, expert_label = training_tokens_and_label(expert_UID_tokenlist,expert_UID_Binary_Label)
    
    #P(c) for expert data
    expert_num_label_test = Counter(expert_label)
    expert_label_prob_test = get_label_prob(expert_num_label_test, 1)
    
    #P(w) for expert data
    expert_word_dict_test = Counter(get_all(set_label_as_post(expert_dataset, expert_label)))
    expert_word_prob_test = get_word_prob(expert_word_dict_test) 
    
    #P(w,c) for expert data
    expert_d_word_dict_test = Counter(get_d_posts(set_label_as_post(expert_dataset, expert_label)))
    expert_word_prob_test = get_word_label_prob(expert_d_word_dict_test, expert_word_dict_test)
    #PMI score for expert data
    expert_PMI_test = get_PMI_score(expert_word_prob_test, expert_word_prob_test, expert_label_prob_test)
    expert_PMI_test = user_pmi_tokens(expert_dataset, list(expert_PMI_test.keys()))
    TaskA_PMI_expert = vect.transform(expert_PMI_test)
    TaskA_PMI_pred_expert = clf.predict(TaskA_PMI_expert)

    print('Task A Expert Evaluation Accuracy on PMI based features: %.3f' %  clf.score(TaskA_PMI_expert , expert_label))
    print(classification_report(expert_label, expert_pred))
    print("Expert Evaluation Confusion Matrix: \n")
    print(confusion_matrix(expert_label, TaskA_PMI_pred_expert))
    plot_confusion_matrix(clf, TaskA_PMI_expert , expert_label, display_labels = label)
    plt.title('Confusion matrix of the PMI SGD Expert Evaluation')
    plt.show()






    

if __name__ == "__main__":
    main()