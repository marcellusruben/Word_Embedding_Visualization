# Word Embedding Visualization

This is a repo to visualize the word embedding in 2D or 3D with either Principal Component Analysis (PCA) or t-Distributed Stochastic Neighbor Embedding (t-SNE).

Below is the snapshot of the web app to visualize the word embedding.

<p align="center">
  <img width="700" height="350" src=https://github.com/marcellusruben/Word_Embedding_Visualization/blob/master/word_embedding_gif.gif>
</p>

## Files

- train_model.py: Python file to load the pre-trained GloVe word embedding model.
- app.py: Python file to create the word embedding visualization web app.
- glove2word2vec_model.sav: Saved pre-trained word embedding model.

To execute the web app, go to the working directory of the app.py and type the following command in the conda environment:
```
streamlit run app.py
```
