import re
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# Text pre-processing for firm1-wise concatenated patent files
# Algorithm:
# 1. Remove punctuations, lowercase text, remove additional whitespaces
# 2. Lemmatise text
# 3. Eliminate stopwords
# 4. Find frequency distribution
# 5. Eliminate corpus specific stop words, words below threshold

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = list(stopwords.words('english'))

# Refining stop words ..
extended = ['least', 'include', 'also', 'one', 'thereof','invention',
            'inc', 'we', 'also', 'us', 'our', 'recent', 'development', 'style', 'margin',
            'top', 'px', 'margin', 'bottom', 'px',
            'left', 'number']

stop_words.extend(extended)


def pre_process_data(text_corpus):
    print('\tProcessing text documents .. ')
    # Algorithm to clean the text.
    # 1. Remove newline characters.
    # 2. Remove punctuation., special characters, numbers.
    # 3. Remove multiple white spaces.
    for index, doc in enumerate(text_corpus):
        # doc = str(doc).lower()
        doc = re.sub('\n', ' ', doc)
        doc = re.sub(r'[^a-zA-Z]', ' ', doc)  # remove everything except words and whitespaces.
        doc = re.sub(r'[0-9]', ' ', doc)
        doc = doc.lstrip()
        doc = doc.rstrip()
        doc = re.sub(r'\s+', ' ', doc)

        text_corpus[index] = doc
        if index % 1000000 == 0 and index !=0:
            print('\tCheckpoint: Complete preprocessing for {}/{}'.format(index, len(text_corpus)))

    print('\tPreprocessing Completed .. ')

    return text_corpus


allowed_pos = [
    'NN',  # noun, singular 'desk'
    'NNS',  # noun plural	'desks'
    'NNP',  # proper noun, singular	'Harrison'
    'NNPS',  # proper noun, plural	'Americans'
    'VB',  # verb, base form	take
    'VBD',  # verb, past tense	took
    'VBG',  # verb, gerund/present participle	taking
    'VBN',  # verb, past participle	taken
    'VBP',  # verb, sing. present, non-3d	take
    'VBZ',  # verb, 3rd person sing. present	takes
    # 'RB',  #adverb	very, silently,
    #'RBR',  #adverb, comparative	better
    #'RBS',  #adverb, superlative	best
    #'RP',  # particle -- give up
    #'JJ',	# adjective	'big'
    #'JJR',	#adjective, comparative	'bigger'
    #'JJS',	#adjective, superlative	'biggest'
    'FW'  # foreign word
]


# Convert standard pos_tag to custom (before stemming/lemmatization)
# Used internally
# NNP, NNS --> n
# V -> v (verb)
# J -> a (adjective)
def pos_tagging(tag):
    if tag.startswith('V'):  # Verb
        return 'v'

    elif tag.startswith('J'):  # Adjective
        return 'a'

    elif tag.startswith('R'):  # Adverb
        return 'r'

    else:
        return 'n'  # Everything else is treated as a noun



# Lemmatization using the standard Wordnet lemmatizer
def lemmatise_text(text_corpus):
    print('\tUsing Wordnet Lemmatization .. ')
    n = len(text_corpus)

    for i in range(len(text_corpus)):
        doc = re.sub('\n', ' ', text_corpus[i])
        doc = re.sub(r'\s+', ' ', doc)
        words = doc.split(' ')
        words = list(filter(None, words))
        tagged_words = pos_tag(words)
        newtext = []

        for j in range(len(words)):
            word = tagged_words[j][0]
            tag = tagged_words[j][1]

            # Lower case after tagging ..
            word = word.lower()
            if word and word not in stop_words and len(word) > 3 and tag in allowed_pos:
                new_word = lemmatizer.lemmatize(word, pos_tagging(tag))
                newtext.append(new_word)

        # Remove punctuations, special characters ..
        newtext = list(filter(None, newtext))
        newtext = ' '.join(newtext)
        newtext = re.sub('[^a-zA-Z]', ' ', newtext)
        newtext = re.sub('\s+', ' ', newtext)

        newtext = newtext.split(' ')
        newtext = [i for i in newtext if len(i) > 2]
        newtext = ' '.join(newtext)
        newtext = newtext.rstrip()
        newtext = newtext.lstrip()

        text_corpus[i] = newtext
        if i % 500000 == 0 and i!=0:
            print('\tCheckpoint: Completed processing {}/{}'.format(i, n))

    return text_corpus


# Calculates the frequency distribution of all unique words across the corpus
def frequency_distribution(text_corpus):
    print('Calculating word frequency')
    word_dict = {}
    i = 0

    for text in text_corpus:
        for word in str(text).split(' '):
            if word_dict.get(word) is None:
                word_dict[word] = 0

            word_dict[word] += 1

        i += 1
        if i % 500000 == 0 and i != 0:
            print(i, len(text_corpus))

    sorted_word_dict={w: freq for w, freq in sorted(word_dict.items(), key=lambda item: item[1])}

    return sorted_word_dict


# Eliminate words word below a user defined threshold
# Eliminate words specific to the corpus
def eliminate_stop_words(text_corpus, dictionary, corpus_stop_words, threshold):
    print('Eliminate corpus-specific stop-words')
    for i in range(len(text_corpus)):
        doc=text_corpus[i]
        words=doc.split(' ')
        newwords=[]

        for w in words:
            if w in corpus_stop_words or dictionary.get(w)<=threshold:
                continue
            newwords.append(w)

        newwords=' '.join(newwords)

        if i%1000000==0 and i!=0:
            print(i,len(text_corpus))

        text_corpus[i]=newwords

    return text_corpus
