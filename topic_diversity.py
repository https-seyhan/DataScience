def topic_diversity(top_words_per_topic):
    all_words = [word for topic in top_words_per_topic for word in topic]
    unique_words = set(all_words)
    return len(unique_words) / len(all_words)

# Usage
topic_diversity_score = topic_diversity(top_words)
