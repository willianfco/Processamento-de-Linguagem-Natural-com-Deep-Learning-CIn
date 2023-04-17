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

The processed dataset will be used on the SVM approaches, since it reduces context dimensionality and its best suited for machine learning technics. On Transformer based approaches we will be using their own pre-process functions, available in HugginFace, which does not remove the words we removed in `pre_process.ipynb` and allows models to be more context aware. 

## Application of NLP Techniques

Since our dataset is imbalanced (≅ 81% Recomendations), we will be using `RandomUnderSampler` from the Imbalanced-Learn library to reduce the amount of positive reviews randomly at our training sets, getting perfectly balanced reviews (50% Recomendations). The Validation, when used as model assessment, and Test sets were not balanced.

   First approach: SVM with Bag of Words

   We joined the train and validation sets to performed a cross-validation using Scikit-Learn SVC and Pipelines, in this case we got 59248 training samples and 17324 test samples. In our tunning tests, we found out that the Sigmoid kernel has proven to converge faster and archieve better performance then linear and rbf for this classification problem. Here are the hyperparameters used to train this model:

      * kernel="sigmoid"
      * tol=1e-3
      * max_iter=-1
   
   With this parameters, using fixed `seed = 0`, the final model performed worse then our naive baseline (≅ 81%) on the test set. Here are the metrics on the hold out data:

      * Accuracy: 0.66
      * Precision: 0.88
      * Recall: 0.68
      * F1: 0.77
      * ROC AUC: 0.65

   Second approach: SVM with Embeddings (Word2Vec)

   Again, we joined the train and validation sets to performed a cross-validation using Scikit-Learn SVC and Pipelines, in this case we got 59248 training samples and 17324 test samples. In our tunning tests, we also found out that the rbf kernel has proven better then any other options for this classification problem. Here are the hyperparameters used to train this model:

      * kernel="rbf"
      * tol=1e-3
      * max_iter=-1

   With this parameters, using fixed `seed = 0`, the final model also performed worse then our naive baseline (≅ 81%) on the test set. Here are the metrics on the hold out data:

      * Accuracy: 0.58
      * Precision: 0.93
      * Recall: 0.52
      * F1: 0.67
      * ROC AUC: 0.68

   Third approach: Bidirectional Encoder Representations from Transformers (BERT)

   For this approach we did not joined train and validation sets, since we were not using cross-validation methods due to model complexity and high computacional costs. The balanced training set had 46178 reviews, validation and test sets had 25985 reviews, a bit more then the last approaches. In this case, we fine-tunned the pre-trained tokenizer and bert-configs from the `bert-base-cased` version available at HuggingFace. The tunning process used thie following hyperparameters:

      * optimizer="Adam"
      * learning_rate=2e-5
      * loss="sparse_categorical_crossentropy"
      * epochs=3,
      * batch_size=16

   With this parameters, using fixed `seed = 0`, the final model performed better then the naive baseline and showed substacial increase on the hability to classify the reviews even with just 3 epochs of tunning. Here are the metrics on the hold out set:

      * Accuracy: 0.86
      * Precision: 0.96
      * Recall: 0.86
      * F1: 0.91
      * ROC AUC: 0.93
   
   Forth approach: Generative Pre-Trained Transformers (GPT-2)

   In this approach, since it is a generative model, it is not suited directly for classifications, but we concatenated the review text with the labels so that the model could learn the contexts in which the review was recomening or not the game. We followed the same steps as we did in the configurations of BERT, but in this case we used `GPT2Tokenizer` and the pre-trained version of `gpt-2` also available at HuggingFace transformers library. 

   Since this is a huge model, it took almost 3 hours to perform a single epoch of training, using `AdamW optmizer` with a realy small learning-rate (5e-4) with a linear schedule with warmupbut. With just this epoch, the model was able to generate reviews that were not present on any dataset and classify them in a reasonable way.

   You can find all codes in this repository, but since the deep learning models were large files, we could'nt load it at our git. If you'd like to have acess to the pre-trained weights of the deep learning methods, contact us. 

   We also performed in-context learn with prompt methods using GPT-4, you can find the results on `prompt_incontext.txt` and in the video presentation of this work.

## Results and Conclusion

Deep learning methods with transfer-learn were able to perform much better then the other methods and archieve a consistently good result using both BERT and GPT-2/GPT-4(prompt), which was a expected result since this pre-trained models have much more parameters and are trained in much more tokens to perform better. 

## References

* [Steam API Documentation](https://partner.steamgames.com/doc/store/getreviews)
* [Natural Language Toolkit (NLTK)](https://www.nltk.org/)
* [SpaCy](https://spacy.io/)
* [Scikit-Learn SVM](https://scikit-learn.org/stable/modules/svm.html#classification)
* [Imbalanced-Learn](https://imbalanced-learn.org/stable/)
* [Tensorflow](https://www.tensorflow.org/text/tutorials/classify_text_with_bert?hl=pt-br)
* [HuggingFace](https://huggingface.co/gpt2)

## License

The project is available under the Creative Commons Zero v1.0 Universal license, which waives copyright interest in a work and dedicates it to the global public domain. Under this license, commercial use, private use, modification, and distribution of the code provided here are allowed. For more information, check in LICENSE
