# selfa-logical-toy
we want to figuere out if self attention can discover simple logical dependencies and explain it correctly. So what we do is we generate sequence of random touples of integers. Whenever a combination of particular touples appears we label it a 1 and if not a zero. This has many analogies in real life. 

We use a standard network architecture of Embedding, Feature extraction, Aggregation and Fully Connected. As the aggregation mechanism we use Attention. The hope is that this attention mechanism should make this model explainable. We compare two different feature extraction methods: Self-Attnetion and Piece Wise Feed Forward networks. Note RNNs or CNNs are not fit for this task ,because the ordering of the input elements is arbitrary. 

For now we use the Attention weights in the aggregation layer for interpretability. We note, that both model are able to perfectly predict on a test set.The Attention weights of Piece Wise Feed Forward Networks are able to perfectly recover the relevant inputs. While the Attention weights of the model trained with Self Attention extraction recover the relevant inputs only very unreliably. 
This cast doubt on the ability of Attention weights to explain Self Attention networks at least for the problem of Sequence Classification. 
