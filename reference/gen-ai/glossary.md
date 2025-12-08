# Glossary

## Neural Networks

*Non-Linearity*

The ability of the model to capture and represent complex relationships in data that cannot be described by a straight line or simple equation.

This allows the network to learn patterns, such as curves or interactions, which are essential for tasks like image recognition and NLP.

*Neuron*

Basic processing unit of a NN. Each neuron takes a feature vector, multiplies it by a corresponding set of weights, adds a bias, and passes the summed results through an activation function (to produce nonlinearity).

*Layer*

A vector of neurons that process info together. Organized in layers (input, hidden, output). 

*Activation function (σ)*

A math function applied to an NN's output. Introduces nonlinearity.

Examples: sigmoid, ReLU, softmax

## Convolutional neural networks (CNNs)

Specialized NNs that process certain forms of structured data (images, video, speech).

*Convolutional layer*

Applies filters to input data using convolutional operations.

*Convolution operation*

Mathematical process for extracting features from data.

It works by: 

- sliding a small filter (a matrix of numbers) over the input (like an image)
- multiplying the filter’s values with the corresponding values in the input and
- summarizing them to create a new, simplified representation of the input.

*Pooling layer*

Reduces the spatial dimensions of data. Decreases computation and overfitting.

This is important because it helps the model handle small changes to inputs (e.g. a slight image rotation or cropping).

Common Methods:

- *max pooling:* retains the maximum value in a region
- *average pooling*: averages values in a region

*Fully connected layer*

Connects every neuron in one layer to every neuron in the next layer, combining extracted features (e.g. nose, eye) to make predictions.

Output is commonly passed through an activation function afterwards.

## Recurrent neural network (RNN)

RNNs are designed to process sequential data by maintaining a memory of previous inputs.

They are useful for handling time-series data or similarly sequential input.

They contain a looping mechanism that computes an internal state update with each time step.

Considered obsolete for many use cases, as transformers have become a near-ubiquitous replacement.

## Transformers

Handle sequential data, including text. 

They use a self-attention mechanism to capture the relationship between words in a sequence.

Their multi-head attention mechanism enhances the model’s capacity to capture diverse patterns, relationships, and context in the input.

*Tokenization*

Converts each word into a fixed-length vector. The resulting vectors are then converted into numerical embeddings.

*Positional encoding*

Captures the significance of word order, as well as the spatial info of each word.

For example, "The sun sets behind the mountain" and "The mountain sets behind the sun" would have the same representation without positional encoding.

*Attention mechanism*

Captures long-range dependencies and represents each token based on its relationship with other tokens.

Example: 

"She poured milk from the jug into the glass until it was full." 

The attention mechanism tells the model what "it" refers to.

*Self-attention*

Computes the importance of different words in a single sequence with each other.

*Multi-head attention*

Allows the model to capture different aspects or patterns in the relationships between words.

It's kind of like running self-attention multiple times in parallel.

Inputs are transformed into subsets and processed (separately in parallel) in self-attention blocks (called "heads").

The result from each head is combined into a matrix. Each head represents an aspect of the relationship between tokens (like positional relationship, syntactic dependencies, or semantic relevance).

Finally, the result matrix is multiplied by a weight matrix. This final matrix is context-enriched representation of the input sequence.

*Cross-Attention*

Allows one set of data (query) to focus on and relate to another set of data (key-value pair). 

It’s like highlighting the parts of one conversation most relevant to the other, ensuring the two sides make sense together.

## Evaluation

*Mode Collapse*

When a model produces repetitive or low-diversity outputs.

### Automatic metrics

Automatic metrics rely on computational methods to assess generative AI outputs.

*Inception score*

Uses a pretrained classifier to see how well it can recognize objects in the generated images. 

The generated images are considered of high quality and diverse if the classifier is confident and finds various objects.

A higher IS indicates that the images are both diverse and realistic.

However, IS assumes that the classifier aligns well with the dataset, which can limit its applicability.

See: [Inception V4 and Xception](https://en.wikipedia.org/wiki/Inception_(deep_learning_architecture))

*Fréchet inception distance (FID)*

Compares the feature embeddings from real and generated images.

The embeddings come from a pre-trained model like Inception V4.

More sensitive to mode collapse than Inception Score.

Lower FID scores mean more similarity between the two distributions, i.e., better images.

*Bilingual evaluation understudy (BLEU)*

Measures the similarity between generated and reference texts by comparing n-grams.

Widely used for translation tasks. Poor for creative tasks.

Ranges from 0 to 1. The higher the better.

*n-grams*

A sequence of words/tokens in text.

1-grams (unigrams) are single words. 2-grams (bigrams) are word pairs.

*Recall-Oriented Understudy for Gisting Evaluation (ROUGE)*

Focuses on recall. Compares how much reference text is included in generated text.

Very effective for summarization tasks. 

Scored from 0 to 1; the higher the better.

Variants:

- ROUGE-N: compares n-grams
- ROUGE-L: measures longest common subsequence (LCS)
- ROUGE-S: measures skip-bigram (bi-gram with at most one intervening word) overlap

*Perplexity score*

Measures how well a language model predicts a sequence of words.

Lower perplexity means greater fluency and confidence in generation.

- Low perplexity (close to 1): The model is confident that it can accurately create the sequence of words.
- High perplexity (>20): The model is unsure.

Example:

> If we have a trained model that has the vocabulary ["Hello", "Cat", "my", "Dog", "name", "is","Edward"], and we want to figure out the perplexity of generating the string: "My name is Edward,"
>
> P("My name is Edward") = P("My") * P("name" ∣ "My") * P("is" ∣ "My Name") * P("Edward" ∣ "My Name is")
> P("My name is Edward") = 0.3 * 0.5 * 0.9 * 0.7 = 0.0945
>
> The perplexity PPL(x) will be 1/PNorm(x)
>
> where PNorm(x) is the normalized probability of the output sentence having a word count of n (n = len(x)),

* Contrastive Language–Image Pre-training Score (CLIP score)*

Evaluates alignment between text and images.

Uses embeddings generated by OpenAI’s open-source CLIP model.

Used to evaluate text-to-image or multimodal generation.

A higher CLIP score means better semantic matching between the text and the image.

- Higher scores (~0.6–1): There is strong alignment.
- Lower scores (0.5 or less): There is weak alignment or unrelated outputs.

### Human Evaluation

For human eval, it is ideal to have a large sample size and a large number of evaluators to reduce bias.

*Mean opinion score (MOS)*

Human evaluators rate outputs on a scale (e.g., 1–5), where higher scores indicate better performance. 

The mean of all scores provides a quantitative assessment of quality.

Widely used, especially to evaluate speech synthesis, text generation, and image quality.

*Task-specific quality evaluation (TSQE)*

Humans assess specific dimensions of quality.

Example:

> In text generation, we may have the following tasks:
> Fluency: Determines the grammatical correctness and natural flow of text.
> Relevance: Determines how well the generated output aligns with the input prompt or context.
> Creativity: Determines the originality and novelty of the content, especially in tasks like story generation or art creation.

*Pairwise comparison*

Comparing two outputs for a given input


























