# Improving Commonsense Validation Using External Knowledge

## Introduction

This is the source code of our final PJ implementation. The best performance reaches 87.80% accuracy on testset.

We borrow the  source code (openKE)  from https://github.com/thunlp/OpenKE to implement KGE techniques on our knowledge base.

We use KagNet's neural network from  https://github.com/INK-USC/KagNet  and re-implement their method on our dataset. We also modify their neural network into our model structure. The modified codes of our model are graph_models.py、graph_train.py


## Files in the folder
```
│  accuracy.py
│  augmentation_FT.py
│  bert_finetuning.py
│  bert_perplexity.py
│  convert_to_statements.py
│  dataloader.py
│  graph_dataset.py
│  graph_gen_cpt.py
│  graph_gen_data.py
│  graph_models.py
│  graph_train.py
│  relation_extraction.py
│  TransE.py
│  trigram.py
│  xlnet_finetuning.py 
|
├─OpenKE
│          
├─perplexity_tf_version
       bert_perplexity_tf.py
       modeling.py
       tokenization.py
      

```

## Running the code

#### 1. Implement validation based on sentence perplexity

```
python bert_perplexity.py
```
We also share a tensorflow version of this implementation. We find it at https://github.com/xu-song/bert-as-language-model.  We test their code on our dataset and get results similar to ours.

#### 2. Implement validation based on fine-tuning
We use two pre-trained language model: BERT and XLNet. 
```
python bert_finetuning.py
python xlnet_finetuning.py
```
#### 3. Utilize external knowledge

1) We firstly obtain ConceptNet from http://www.conceptnet.io/ and clean the original knowledge base. Then we implement Trans-E on the cleaned knowledge base.

2) We preprocess our dataset

3) We extract and save relations from knowledge base for dataset and generate schema graph

```
python graph_gen_cpt.py
python TransE.py sgd 0

python dataloader.py
python convert_to_statements.py

python relation_extraction.py
python graph_gen_data.py
```
#### 4. Augmenting fine-tuning with external knowledge

We convert retrieved knowledge into natural language  and add it to our dataset. Then we implement fine-tuning again.

```
python augmentation_FT.py
python bert_finetuning.py
python xlnet_finetuning.py
```
#### 5. Implement our model and generate result file

Overall Workflow 
<img src="overallframework.png" style="zoom:50%;" />

We test our final model on our dataset and generate a csv file as our final result.

```
python graph_train.py
python accuracy.py
```