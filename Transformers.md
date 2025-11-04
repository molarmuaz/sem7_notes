# Transformers
## Paper: Attention is all you need

### Why Transformers
Sequence Transduction i.e. the process of converting a sequence into another sequence like prompt to response, translating from a language to another, speech recognition, video captioning was mostly done with RNNs, LSTMs, or GRUs. 

All these models processed the input step by step or token by token and kept a hidden state that carried the context through the steps. This method provides a huge bottleneck that slowed down this process heavily. Its hard to parallelize them and even then long term context can still be lost. CNN layers were implemented to capture patterns in parallel. They still struggled with long range dependencies. 

Transformers use an attention mechanism instead to **directly** connect any two tokens in a sequence regardless of how apart the tokens are. Now we are able to feed entire sentences and have it be processed in parallel. Each token's vector also depends on its context so we had better understanding of data with transformers and we could distinguish river bank from the financial one by "paying attention" to the context.

Pros:
- Faster
- Easier to Scale
- Captures long term contexts

### Intro
Attention based deep learning seq2seq model based on encoder-decoder architecture

Well known models based on transformers:
- BERT - Google 
- GPT - OpenAI
- RoBERTa - Facebook AI

#### Attention 
Global or soft attention helps the model to focus on relevant parts of input sentence. The attention mechanism assigns **weights** to input tokens, showing how important each word is relative to others for the current context or query. These weights are usually computed from the **similarity** between token embeddings (like how much one word should “pay attention” to another). The model then takes a **weighted sum** of all token representations using these attention scores, producing a context-aware output vector that reflects meaning based on relationships in the sequence.

##### Self-Attention
Also known as scaled dot-product attention. Each element in a sequence computes its attention score with respect to **every other** element in the same sequence, including itself. It calculates weights for every word in a sentence as they relate to every other word in the sentence, so the model can predict words which are likely to be used in sequence.

Since attention lets every token look at all others **simultaneously**, there’s no need to process tokens one by one like an RNN. You feed the entire sequence into the model, and it computes all the pairwise attention scores (the dot products between queries and keys) **in parallel** using matrix operations.

Models can “learn” the rules of grammar, based on statistical probabilities of how words are typically used in language.

### Architecture
Consists of encoders and decoders in the original paper, but depending on need only one of them can be structured as well. 

Encoder converts our input sequence x into vectors z, the decoder converts our vectors z into another sequence y.

Each encoder/decoder consists of stacked self attention layers and fully connected layers.

Let's decode all of the components of this diagram
![[attachment/Transformers/file-.png]]

#### Encoder

![[attachment/Transformers/file- 1.png]]
##### Input Embeddings
Converting our input tokens to *embeddings.*

> **Embeddings:** Dense vector representations typically of fixed length represented by floating point numbers

we can use a model like word2vec for generating these or learn from scratch using a neural net of our own or use contextual word embeddings with models like BERT or ELMo

##### Positional Encoding
Since we are not using any **recurrence** or **convolution** we need some way to know the position the tokens are sequenced for e.g in a sentence like "Kanye has delayed his album for the fifth time" we want to know 'Kanye' came before 'has' and so on. 

For this we use positional encoding to assign a unique representation to each token with respect to their location in a sequence. *The size of these positional embeddings is the **same** as the vector embeddings of our tokens*

###### Formula
%% Let's say our model takes vector embeddings of size 6. %%
d_model = dimension of vector embedding,
i = index within the vector
pos = position of the word in our sentence

> Our even numbered indices are calculated by a sine function and odd numbered indices are calculated with a cosine function

![[attachment/Transformers/file- 2.png]]

for example, the word album in our sentence  "Kanye has delayed his album for the fifth time" would have:

pos = 4 (first word is 0)
d_model = 6 (as mentioned above)

our i goes from 0 to d_model/2 - 1, since they are divided between sine and cosine pairs, so run 0, 1, 2 between both

i = 0:
	PE(4,0) = sin(4/1) ≈ −0.**7568**
	PE(4,1) = cos(4/1) ≈ −0.6536
i = 1:
	PE(4,2) = sin(4/10000^2/6) ≈ 0.1845
	PE(4,3)=cos(4/21.5443) ≈ 0.9827
i = 2:
	PE(4,4)=sin(4/464.159) ≈ 0.008618
	PE(4,5)=cos(4/464.159) ≈ 0.99996
So, the word 'album' would have a positional encoding of [−0.7568,−0.6536,0.1845,0.9827,0.008618,0.99996]

##### Attention Mechanism
With this we can assign different attention scores to different tokens to so we can:
	Give more importance to relevant information,
	Ignore irrelevant information,
	Effectively capture long-range dependencies

Self-attention takes **Q (queries), K (keys), and V (values)**. Q is compared to K to see how important each word is to every other word. Then we do a **weighted sum of the V’s**.

Q, K, and V come from the **word embeddings**. Each embedding is passed through **three small learned linear layers**. The weights of these layers are learned during training. The outputs are the Q, K, and V vectors.

Since Q and K have the same dimension, we compute **Q·Kᵀ**. This gives **similarity scores** between words. Then we divide by **√(dimension)** to scale it so softmax does not vanish or explode. **Softmax** turns the scores into probabilities that sum to one.

Finally, we multiply the softmax weights by V to get the self-attention output:

$$ 
\text{Attention}(Q, K, V) = \text{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg) V  
$$
There's also a dot-product attention which doesn't use scaling which makes it faster and uses less space, and an additive addition attention that feeds the value through a one hidden layer network.

![[attachment/Transformers/file- 3.png]]

**Self-attention:** Each word looks at itself and its neighbors to understand the sentence, like **a student reviewing their own notes and highlighting connections between topics**.

**Cross-attention:** One sequence looks at another sequence for guidance, like **a translator reading a source text while writing in their own language**.

###### Multi-head attention
Multi-head attention extends the idea of self-attention by allowing the model to focus on multiple parts of the input sequence simultaneously. Instead of running a single attention function, the model performs several attention computations in parallel. Each one called a “head.”

In multi-head attention, the process starts by deciding how many heads the model will use. Each head has its own set of learned linear layers that project the same input embeddings into separate queries, keys, and values. 

Every head then performs self-attention independently on its own Q, K, and V, learning to focus on different parts or relationships within the sequence. Once all heads have produced their output context vectors, those outputs are concatenated into a single matrix. Finally, this combined result is passed through another linear layer that mixes information from all heads into one unified representation.

![[attachment/Transformers/file- 4.png]]
##### Feed-forward Layer
After the attention step, every token now carries information about how it relates to other tokens in the sequence. But attention mostly handles _relationships between tokens_. What it doesn’t do is refine each token’s internal meaning. That’s where the **feedforward layer** comes in.

The feedforward layer is just a small neural network applied separately to each token’s vector. It first expands the vector to a higher dimension, applies a non-linear activation like ReLU or GELU, and then compresses it back to the original size.

This process helps the model learn more complex transformations and build deeper representations of each token. So while attention lets tokens “talk” to each other, the feedforward layer lets each token “think” on its own before moving to the next layer.

After both the multi-head attention layer and the feed-forward layer, an add and norm layer is attached
##### Add & Norm
**Residual Connection (Add):**
After we have our results from the attention layer, we add the original input into the output. This prevents the vanishing gradient and keeps the original information flowing through the network.

**Layer Normalization:**
We also normalize the values after adding which basically takes each layer of our answer (input + output) and scale it to have a mean of 0 and a variance of 1 for example so we have smooth flowing consistent gradients to avoid vanishing gradients and to make computation easier

#### Decoder
As we can see from the diagram, the decoder is pretty much the same as the encoder except a few nuances.

##### Output Embeddings
Since now we want the decoder to *predict* the tokens, we shift all token to the right and add an [SOS] (start of sequence) token at the start that is just a cue for the decoder to start predicting, so out previous input would become  "[SOS] Kanye has delayed his album for the fifth time".

##### Linear Layer
Before talking about the masked multi-head layer let's talk about how the output is generated. At the end of the decoder, there’s a **linear layer**. The decoder outputs a vector representing what it “wants to say next.” The linear layer compares this vector to all word embeddings in the vocabulary and finds which one is closest in meaning. Then, a softmax converts those similarity scores into probabilities, and the model picks the word with the highest one.

This process repeats for every token each time the model re-evaluates the entire vocabulary to find the next word. It’s powerful but computationally heavy and one of the biggest bottlenecks in LLMs.

**Extra Info:** The temperature setting we see in setting up LLMs is a number we divide the results with before softmax so smaller temp score amplifies probabilities making it more confident on the words to pick. Bigger temp score shrinks all probabilities making the model uncertain over which word to pick making it more uncertain and random/creative.
##### Masked Multi-head Attention
After the output word is generated, it is appended to the input.

Similar to the multi-head attention layer we use normally in both the encoder and decoder, the masked multi-head attention layer is set so that it prevents the decoder from seeing into the future. This masking makes sure that during training predictions depend only on previous tokens and not to the ground truth.

During training we use this to compare how accurate our predicted word is. Together, this setup lets the decoder learn to generate text one word at a time while still being trained efficiently in parallel.

