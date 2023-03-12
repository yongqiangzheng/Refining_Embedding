# Refining_Embedding

![model](/Model.png)

## Introduction
Refine GloVe word embedding by Vader lexicon to learn sentiment information.

## Usage
Download GloVe [here](https://nlp.stanford.edu/data/glove.840B.300d.zip)

```
python -m venv .venv
source .venv/bin/acitvate
sh run.sh
```

## t-SNE visualization

### good
![fig1](/result/good.png)

### bad
![fig2](/result/bad.png)

### fortunate
![fig3](/result/fortunate.png)

### unfortunate
![fig4](/result/unfortunate.png)

## Reference

Yu, L. C., Wang, J., Lai, K. R., & Zhang, X. (2017). Refining word embeddings using intensity scores for sentiment analysis. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 26(3), 671-681.

Pennington, J., Socher, R., & Manning, C. D. (2014, October). Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) (pp. 1532-1543).

Hutto, C., & Gilbert, E. (2014, May). Vader: A parsimonious rule-based model for sentiment analysis of social media text. In Proceedings of the international AAAI conference on web and social media (Vol. 8, No. 1, pp. 216-225).