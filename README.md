<div align="center"> <h1><I>deepGraphh</I> </h1> </div>
 <br>
<div align="center">
<img src="Data/Images/gif 5.gif"></div>
<br><br><br>
Artificial intelligence-based computational techniques allow the rapid exploration of the chemical space. Recent algorithmic advancements in modeling neural networks provide an automatic and extraordinary means for molecule representation, in contrast, to traditional manually designed fingerprints or descriptors. Although the implementation of graph-based methods for chemical properties calculation offers multiple advantages, its implementation demands in-depth domain knowledge and programming skills. To address this, here we introduce <I>deepGraphh</I>, an end-to-end web service featuring a conglomerate of powerful graph-based neural networks methods for model generation for classification or regression tasks. The Graphical User Interface of <I>deepGraphh</I> supports highly configurable parameters support for model tuning, generation, and testing of the user-supplied query molecules. <I>deepGraphh</I> supports four widely accepted methods for graph-based model generation i.e., Graph Convolution Network (GCN) (Duvenaud et al., 2015), Graph Attention Network (GAT) (Veličković et al., 2018), Directed Acyclic Graph (DAG) (Ha, Sun and Xie, 2016), and AttentiveFP (Xiong et al., 2020). Importantly, for all the aforementioned methods, <I>deepGraphh</I> also supports cross-validations and returns comprehensive results, both in graphical as well as tabular formats. One of the key highlighting features of deepGraphh is that it allows live tracking of the main steps involved in the model generation. To our knowledge, deepGraphh is the first open-source and multi-functional graph-based deep learning framework supporting web service to date. 
<br><br>

**Webserver is freely available at https://deepgraphh.ahujalab.iiitd.edu.in**

The source code of the embeddings needed to train the model is available at  https://github.com/the-ahuja-lab/deepGraphh

## Prediction Engines:
<div align="center">
<img src="Data/Images/github_ss_deepGraphh.PNG"></div>

## Additional Features:
1. <I>deepGraphh</I> is a one-stop web service for graph-based methods for cheminformatics
2. <I>deepGraphh</I> is one of the only open-source web servers to date that provides multiple GUI-based options for graph-based QSAR analysis. 
3. <I>deepGraphh</I> is backup by significant computational resources to enable model generation on larger datasets.
4. <I>deepGraphh</I> allows users to temporarily store their data for up to 1 month.
5. <I>deepGraphh </I>is open source and free to use.

## How to run pre-trained <I>deepGraphh</I> Models
There are two ways users can use saved models and get the predictions on query data
1. Users can use <b>GetPrediction_From_Checkpoint_deepGraphh.ipynb</b> in the <b>Get Prediction</b> folder.
2. Users can use the link provided in the <b>tutorial section</b> of the <I>deepGraphh</I> web server and it will be directed to google colab. The users need to make a copy of this colab to get their predictions.<br>
<b>All the sections in the python Notebook are well commented so that the user has easy access.</b>

