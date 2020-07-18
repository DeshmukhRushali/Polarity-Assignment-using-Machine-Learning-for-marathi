import nltk
import pickle
import random
from nltk.tokenize import word_tokenize
import seaborn as sns
from nltk.corpus import brown
from nltk.corpus import indian
from mlxtend.evaluate import paired_ttest_kfold_cv
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from nltk import MaxentClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import chi2
import numpy as np
import itertools
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import csv
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
import math
from sklearn.model_selection import learning_curve

from sklearn.model_selection import validation_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_curve, auc

from sklearn.utils import shuffle
import sklearn_crfsuite
import yaml
from sklearn.feature_selection import SelectKBest
import os
def term_frequency(term, tokenized_document):
    return tokenized_document.count(term)

def sublinear_term_frequency(term, tokenized_document):
    count = tokenized_document.count(term)
    if count == 0:
        return 0
    return 1 + math.log(count)
def inverse_document_frequencies(tokenized_documents):
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
    return idf_values

def tfidf(documents):
  #  tokenized_documents = [d for (d,c) in documents]
    idf = inverse_document_frequencies(documents)
    tfidf_documents = []
    
    #labels = [c for (d,c) in documents]
    for document in documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            tid=tf*idf[term]
            doc_tfidf.append(tid)
            #print("term=",term,"tfidf=",tid)
        tfidf_documents.append(doc_tfidf)
        
    return tfidf_documents

def createList(foldername, fulldir = True, suffix=".jpg"):
    file_list_tmp = os.listdir(foldername)
    #print len(file_list_tmp)
    file_list = []
    if fulldir:
        for item in file_list_tmp:
            if item.endswith(suffix):
                file_list.append(os.path.join(foldername, item))
    else:
        for item in file_list_tmp:
            if item.endswith(suffix):
                file_list.append(item)
    return file_list
class Splitter(object):
    def __init__(self):
        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()
        
    def split(self, text):
        """
	input format: a paragraph of text
	output format: a list of lists of words.
	e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
	"""
        sentences = self.nltk_splitter.tokenize(text)
        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
        return tokenized_sentences



def trainDecisionTree(trainFeatures, trainLabels):
    clf = make_pipeline( OneVsRestClassifier(DecisionTreeClassifier(criterion='entropy')))
    
    scores = cross_val_score(clf, trainFeatures, trainLabels, cv=5)
    clf.fit(trainFeatures, trainLabels)
    filename = 'DT_hposmodel.sav'
    pickle.dump(clf, open(filename, 'wb'))
    return clf
    
    #scores = cross_val_score(clf, trainFeatures, trainLabels, cv=5)
    #clf.fit(trainFeatures, trainLabels)
    #filename = 'DT_hposmodel.sav'
    #pickle.dump(clf, open(filename, 'wb'))
    #return clf
def trainRF(trainFeatures,trainLabels):
    clf = make_pipeline(RandomForestClassifier())
    clf.fit(trainFeatures,trainLabels)
    filename = 'RF_hposmodel.sav'
    pickle.dump(clf, open(filename, 'wb'))
    return clf

def trainNaiveBayes(trainFeatures, trainLabels):
    clf = make_pipeline(MultinomialNB())
    scores = cross_val_score(clf, trainFeatures, trainLabels, cv=5)
    clf.fit(trainFeatures, trainLabels)
    filename = 'NB_hposmodel.sav'
    pickle.dump(clf, open(filename, 'wb'))
   # return clf, scores.mean(),scores
    return clf
def trainNN(trainFeatures, trainLabels):
    clf = make_pipeline(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100), random_state=1))
    #scores = cross_val_score(clf, trainFeatures, trainLabels, cv=5)
    clf.fit(trainFeatures, trainLabels)
    filename = 'NN_hposmodel.sav'
    pickle.dump(clf, open(filename, 'wb'))
    return clf

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#marathi_sent = indian.sents('marathi_pos_rad_3NOV17.pos')
#mpos= indian.tagged_sents('marathi_pos_rad_3NOV17.pos')
#mp=shuffle(mpos)


#print("POS tagger")
#size = int(len(marathi_sent) * 0.67)
#tags = [tag for (word, tag) in indian.tagged_words('marathi_pos_rad_3NOV17.pos')]
#print(tags)
#print(np.unique(tags))
#print("no. of tags=",len(nltk.FreqDist(tags)))
#defaultTag = nltk.FreqDist(tags).max()

#print(defaultTag)
#train_sents = mp[:size]
#print(len(train_sents))#test_sents = mp[size:]


file1=createList("D:rad_phd_final_26sept19/sentiment_dataset/POS",suffix=".txt")

file2=createList("D:rad_phd_final_26sept19/sentiment_dataset/NEG",suffix=".txt")

documents=[]
all_words=[]
for fname in file1:
    a=list(nltk.corpus.indian.words(fname))
    b=[]
    for w in a:
        #if w not in word:
        b.append(w)
    documents.extend([(b,"pos")])
    all_words.extend(b)

for fname1 in file2:
    a=list(nltk.corpus.indian.words(fname1))
    b=[]
    for w in a:
        
     #   if w not in word:
        b.append(w)
    documents.extend([(b,"neg")])
    all_words.extend(b)

random.shuffle(documents)
size = int(len(documents) * 0.7)
tags = [tag for (document, tag) in documents]
train_sents = documents[:size]
#print(len(train_sents))
test_sents = documents[size:]
#trainFeatures, trainLabels = transformDataset(train_sents)
#testFeatures, testLabels = transformDataset(test_sents)
corpus=[d for (d,c) in documents]
Label=[c for (d,c) in documents]
Features=tfidf(corpus)
print("Length of features",len(Features))
#test = SelectKBest(score_func=chi2, k=150)
#fit = test.fit(Features,Label)
#print(test.pvalues_)
#print(Label)

var = 1
trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(Features,Label, test_size=0.20, random_state=42, stratify=Label)

while var == 1:
    print("******************MENU********************")
    print("case 1:Naive Bayes classifier")
    print("case 2: Decision tree classifier")
    print("case 3: Neural network")
    print("case 4: K nearest neighbour")
    print("case 5: Random forest classifier")
    print("case 6:exit")   
    print("enter your choice")
    ch=input()
    if ch == "6":
        
        var = 2
        continue
   
    elif ch == "1":
        print("naive bayes")
        tree_model= trainNaiveBayes(trainFeatures, trainLabels)
        NB = tree_model
        
    elif ch =="2":
        tree_model = trainDecisionTree(trainFeatures, trainLabels)
        DT = tree_model
    elif ch == "3":
        tree_model= trainNN(trainFeatures, trainLabels)
        NN = tree_model
    elif ch == "4":
        tree_model = trainkNeighbour(trainFeatures, trainLabels)
        KNN = tree_model
    elif ch == "5":
        tree_model= trainRF(trainFeatures, trainLabels)
        RF = tree_model
    
    Max = 0
  #  for i in range(1,6):
  #      print("accuracy in fold ",i, " =",scores[i-1])
            
            
   # print("accuracy on train data  = ")
   # print(tree_model_cv_score)
    y_pred = tree_model.fit(trainFeatures, trainLabels).predict(testFeatures)
       #print("y_predicted=",y_pred)    
       # print("y_pred unique= ",np.unique(y_pred))
        #print("y_test unque=",np.unique(testLabels))

#scatter plot
#plt.figure()
#plt.scatter(testLabels, y_pred)
#plt.xlabel("True Values")
#plt.ylabel("Predictions")
#plt.show()
#end
    classes=['neg', 'pos']
    cnf_matrix = confusion_matrix(testLabels, y_pred)
    np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=np.unique(tags),title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
    
    #plt.figure()
    #plot_confusion_matrix(cnf_matrix, classes=np.unique(tags), normalize=True,title='Normalized confusion matrix')


    random_state = np.random.RandomState(0)
    classes = ['neg','pos']
    #if(ch!="8"):
    t1=testLabels
    testLab= label_binarize(testLabels,classes)
    print(testLab.shape[1])
    y_p=label_binarize(y_pred,classes )

#print("len of te_lab",len(te_lab))
#print("len of y_pr",len(y_p))

    y_p_new=[]

    #classes = ['neg','pos']
    
    
    
    precision = dict()
    recall = dict()
    ther = dict()
    average_precision = dict()
    print(classification_report(testLab, y_p, target_names=classes))
    precision["micro"], recall["micro"], _ = precision_recall_curve(testLab.ravel(),y_p.ravel())
    average_precision["micro"] = average_precision_score(testLab, y_p,average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))


    
    
            
    cnf_matrix = confusion_matrix(testLabels, y_pred)
    np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=np.unique(tags),title='Confusion matrix, without normalization')





    plt.figure()
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,where='post')
    plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2,color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))
    plt.show()

   

    plt.figure()
    train_sizes, train_scores, validation_scores = learning_curve(tree_model, Features ,Label, train_sizes=np.logspace(-1, 0, 20))
    plt.xlabel('Trainging Examples')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.plot(train_sizes, validation_scores.mean(axis=1), label='cross-validation')
    plt.plot(train_sizes, train_scores.mean(axis=1), label='training')
    plt.legend(loc='best')
    plt.show()

   
