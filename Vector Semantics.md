# Vector Semantics
Humans understand relations and meanings intuitively e.g. cat and dog are related \[animals] or coffee and cup \[usage] 

A question that we battled with for a long time and still kind of do is how to teach these relations and meanings to a computer. 

### Basic Intuitions:
- Since computers only understand numbers, we must have a numeric representation of our words for computers to decode the relations
- It can't be so computationally expensive that we can't encode an entire language.
- The relations must be calculatable numerically (e.g. cosine similarity)
- **Lexical Semantics:** The more two words are used in similar contexts, the more similar they are in meaning e.g. "my pet cat" and "my pet dog".
### Vector Embeddings
It was clear we need vector embeddings i.e. arrays of values that we can use to form relations.

#### Evolution

##### 1. One-hot Encoding
We started with one-hot encoding. Each word's vector is the size of the vocabulary where each index represents a single word in the vocabulary. For each word's vector we set the value of its index to 1 while the rest are 0. 

This had several issues like requiring too much memory since each word was the size of the entire vocabulary. secondly, we can't even use these to find similarity since cosine similarity would always give us 0 or 1. 

**Q. Why not use binary instead of One-hot?**
	One hot separates the identity of each word like cat `100`, dog `010`, fish`001`. With binary for example, cat = `001`, dog = `010`, fish = `011`), the model sees numeric _relationships_ that don’t exist. “Dog” suddenly looks closer to “Fish” than to “Cat” because 010 and 011 differ by one bit.

> While we don't "use" one-hot anymore, our modern day embeddings still stand on one hot encoding vectors trained.

#### 2. Osgood Method
This method proposed manual rating of each word on valence, arousal and dominance. This would've had to be manual and still could turn-out inaccurate. Furthermore, some words would not fit inside the box of these three categories like 'this'.

#### 3. Term-Document Matrix
In this method, we would map occurrence of each term in a document for all documents. This can be useful in very specific cases maybe like genre-classification for books. our vector would be counts of a word being used in each document. We could also get document embeddings by flipping the table. 

This of course had a lot of problems; the count of each word in each document was still kind of raw, even though now we could take cosine similarities, it wasn't very useful since the vectors were not really mapped in a way to show relation between words, just how they appeared. The vectors would still sometimes turn out very sparse, like the word 'ghost' might be mentioned mainly in horror books and nowhere else. With increase of a word all the documents vectors would increase in size or vice versa and recalculation would be needed for the new word in every document.

context not taken into consideration (river bank and financial bank were the same).

**Q. Why sparse embeddings are bad?**
	each word’s vector marks which documents it appears in. “Cat” and “dog” might show up in different ones like cat blogs vs. dog guides. So, their vectors barely overlap. That makes them look unrelated, even though they clearly share meaning. Most entries are zeros anyway, so the math can’t see subtle links. Dense embeddings fix that by learning smaller, continuous vectors where similar words naturally end up closer together. (“Far apart” here just means **orthogonal**; no shared direction.)

#### 4. Term-Term Matrix
In this method, we would map terms against terms counting how many times they occurred together.

Some words like 'the' are frequent and appear everywhere. This makes the model give importance to such words when it shouldn't bias towards such a meaningless word. 

##### TF-IDF (Term Frequency - Inverse Doc Frequency)
TF-IDF attempts to solve the issue of filler words or meaningless words forming a bias in our vectors. We take the ratio of total documents (N) to the number of documents a word occurred in (df<sub>t</sub>). So, a word like the that appears multiple times in every document gets a very negligible value. The rare numbers get a huge spike (a number that occurred once in one document will have a score of N while common words might not even have a score of 1), so we scale them using log. Lastly, we add a 1 in case the frequency = documents and their log comes out 0.


$$\text{IDF}(t) = \log\left(\frac{N}{\text{DF}(t)}+1\right )$$

We multiply the IDF term with our already calculated TF(t , d) to get our TF-IDF

$$ 
\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t)}\right) 
$$

- TF(t , d)=term frequency of term t in document d,    
- N = total number of documents,
-  df<sub>t</sub> = number of documents containing term t.

### Measuring Similarity
We measure similarity between words on two things mainly. 
- **Magnitude** (length): Often represents frequency and strength
- **Direction**: Encodes the pattern of feature value

For example a king and queen might have similar magnitude and direction whereas a hero and a villain might have similar magnitude but opposite directions.  In semantics we care more for direction than magnitude. In reality its not as similar as same direction or opposite directions, rather angles between the vectors' directions.

Since, we normalize, the magnitude bias is removed anyways and similarity is focused on more.

#### Cosine Similarity
The initial idea was to take a dot product, but with more frequent words the dot product would obviously turn out greater, forming an illusion of similarity. To normalize this, we divide the dot product by their magnitudes

![](attachment/Vector%20Semantics/file-.png)

##### Why it works

- If two vectors point in the same direction → angle = 0 degrees, cosine = 1 → very similar.
    
- If they are unrelated → angle = 90 degrees, cosine = 0 → no similarity.
    
- If they point in opposite directions → angle = 180 degrees, cosine = -1 (but in text, vectors are usually non-negative, so cosine values typically range from 0–1).

In text data, opposite vectors almost never show up because word and document vectors are usually **non-negative**. Their components come from counts, probabilities, or activations that don’t dip below zero.

##### Example use

![](attachment/Vector%20Semantics/file-1.png)

#### Pointwise Mutual Information (PMI)
PMI compares the probability of two words occurring together vs. independently.
##### Interpretation:
**High PMI** → the words co-occur more often than expected.
**Low or negative PMI** → the words co-occur less often than expected.

$$
\text{PMI}(x, y) = \log \frac{P(x, y)}{P(x) , P(y)}  
$$

P(x , y) = Probability of x and y to appear together (co-occurrence)
P(x) = Probability of x appearing.

If they co-occur more often than chance: → numerator > denominator → PMI > 0
If they rarely co-occur: numerator < denominator and the log of n<1 is negative so → PMI < 0

##### Example
![](attachment/Vector%20Semantics/file-2.png)


### Terms from slides

**Lemma**: Dictionary base form of a word
	Example: lemma mouse covers both mouse (singular) and mice (plural)
	
**Wordform:** Actual inflected version in text
	Example: sings, sang, sung are wordforms of lemma sing
	
**Word Sense:** One specific meaning of a word
	mouse → sense 1: a rodent
	mouse → sense 2: a computer device
	
**Polysemy:** One lemma with multiple senses
	Example: bank (river bank vs financial bank)
	
**Synonymy:** Two words have nearly identical meaning.
	sofa ≈ couch
	
**Similarity:** Words share many features but aren’t identical.
	cat and dog (both animals, but not synonyms)
	
**Relatedness:** Words often co-occur but aren’t “similar.”
	doctor – scalpel, coffee – cup
	
**Semantic Field:** A group of words belonging to the same theme/domain.
	Hospital domain: surgeon, nurse, anesthetic, patient




