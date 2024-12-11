import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models.ldamodel import LdaModel

# Download required NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

print(f"Il path è  {nltk.data.path}")


def preprocess_text(text):
    """Preprocess the text: tokenization, removing stopwords and non-alphanumeric characters."""
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    # Keep only alphanumeric tokens and remove stopwords
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return tokens

def extract_topics(text, num_topics=3, num_words=5):
    """
    Extract topics from text using LDA.
    Args:
        text (str): Input text.
        num_topics (int): Number of topics to extract.
        num_words (int): Number of words per topic to return.
    Returns:
        list of tuples: Topics represented by their top words.
    """
    # Preprocess the text
    processed_tokens = preprocess_text(text)

    # Create a dictionary and corpus for LDA
    dictionary = corpora.Dictionary([processed_tokens])
    corpus = [dictionary.doc2bow(processed_tokens)]

    # Train the LDA model
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, random_state=42)

    # Extract topics
    topics = lda_model.print_topics(num_topics=num_topics, num_words=num_words)
    clean_topics = []
    for _, topic in topics:
        # Extract words only
        words = []
        for word in topic.split('+'):  # Split by "+"
            word_cleaned = word.split('*')[1].strip("\" ")  # Get the word and remove quotes
            words.append(word_cleaned)
        clean_topics.append(words)
    return clean_topics

# Example Usage
if __name__ == "__main__":
    text = """
    in recognition of the extraordinary services he has rendered by the discovery of the laws of chemical dynamics and osmotic pressure in solutions
    """
    topics = extract_topics(text, num_topics=1, num_words=5)
    for idx, topic in enumerate(topics):
        print(f"Topic {idx + 1}: {','.join(topic)}")
