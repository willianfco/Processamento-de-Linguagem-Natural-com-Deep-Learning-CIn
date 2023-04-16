# NLP Project - Analysis of Dead by Daylight Game Reviews

Authors: Julierme Silva and Willian Oliveira

This project is part of the assessment for the Deep Learning specialization's Natural Language Processing with Deep Learning course at CIn-UFPE. It aims to employ Natural Language Processing (NLP) techniques in analyzing reviews for the Dead by Daylight game on the Steam platform. The primary focus is on sentiment analysis, categorizing `Recommended` reviews as positive and `Not Recommended` reviews as negative.

## Objectives

1. Collect all English reviews of the Dead by Daylight game on the Steam platform using the Steam review API.
2. Perform preprocessing and data cleaning.
3. Apply NLP techniques to perform sentiment analisys and classify reviews using:

   * SVM with Bag of Words
   * SVM with Word2Vec
   * BERT
   * GPT-2

## Requirements

- Python 3.x
- Pandas
- Requests
- NLP Libraries (NLTK, SpaCy, etc.)
- Machine Learning Libraries (Scikit-Learn e Imbalanced-Learn)
- Deep Learning Libraries (TensorFlow 2.x, Torch 1.12.x)

## Installation

To install the required libraries, run the following command:

```python
   pip install -r requirements.txt
```
## Data Collection

The `get_reviews.ipynb` script collects all English reviews of the Dead by Daylight game on the Steam platform. The script uses the Steam review API and saves the collected reviews in a CSV file for further processing.

## Preprocessing and Data Cleaning

The `pre_process.ipynb` script discards blank reviews and performs the following text processing technics:

    1. Remove special characters and lowercase the text
    2. Tokenize the text
    3. Remove stopwords
    4. Lemmatize the words
    5. Reunite the words back to string

After that, the script splits the processed dataset into Train, Validation and Test and save them as numpy zipped archives on the `processed` data folder.

## Application of NLP Techniques

## Results and Conclusion

## References

* [Steam API Documentation](https://partner.steamgames.com/doc/store/getreviews)
* [Natural Language Toolkit (NLTK)](https://www.nltk.org/)
* [SpaCy](https://spacy.io/)
* [Scikit-Learn SVM](https://scikit-learn.org/stable/modules/svm.html#classification)
* [Imbalanced-Learn](https://imbalanced-learn.org/stable/)
* [Tensorflow](https://www.tensorflow.org/text/tutorials/classify_text_with_bert?hl=pt-br)
* [HuggingFace](https://huggingface.co/gpt2)

## License


