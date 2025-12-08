# Text2Text Generation Systems

Conversational LLMs are specifically trained to engage in interactive dialogue.

Use Cases:

- Chatbots
- Virtual assistants
- Interactive storytelling

## Requirements

*Functional*

- *Natural language understanding:* identifying entities, intent, and sentiment from user input
- *Content retention:* storing and recall details from earlier in the conversation
- *State management:* tracking conversation flow, unresolved queries, and user end goals
- *Personalization:* tailoring responses based on user preferences and historical interaction
- *Natural language generation:* accurate query responses with naturala language

*Non-Functional*

- Low latency
- Scalability
- Availability
- Reliability
- Security (safeguarding, PII protection, prevention of prompt jailbreaking & prompt injection)

Note: There are tradeoffs between latency and accuracy.

## Model Selection

A good starting point is the the Llama 3.2 3B model (rather than the larger 11B or 90B models).

It is open-source, with both small size and good accuracy. 

Compared to larger models, it has faster training, reduced computational costs, faster inference times, and easier deployment.

## Training Infrastructure

### Training process

*Data Prep*

Steps:

1. Collect the dataset
2. Preprocess data (removing offensive/irrelevant content, ensuring consistency)
3. Store the processed dataset

Training Sources:

We can aggregate data from multiple sources. There are multiple publicly available datasets.

Assume the complete dataset is 50GB with 200M rows of text.

Format:

| Prompt/Query | Response |
| -----------  | -------- |
| What's the capital of Illinois? | The capital of Illinois is Springfield. |
| Is pluto a planet? | That’s a great question — and the answer depends on whether you’re asking from a historical, scientific, or cultural perspective.<br /><br /> Here’s a breakdown:<br/>...|

Note: Many off-the-shelf datasets will come in formats that may not be suitable for training immediately. We may need to remove or merge some columns (features) to prepare the dataset for training.

*Preprocessing Considerations*

- **Topics:** the distribution of topics should ensure balanced representation. Some topics (weather, hobbies) may appear more frequently in the dataset. This could lead to model bias.
- **Demographics:** biases towards sex, race, age, or socioeconomic condition. 
- **Sentiment:** bias toward positive/negative sentiment
- **Noise Removal:** like HTML tags, emojis, or weird punctuation
- **Rare Word Substitution**
- **Formatting**
- **Special characters and cases**

Human review may be needed.

*Bias Mitigation*

- **Synthetic data generation:** can address underrepresented topics or demographics
- **Downsample overrepresented groups**
- **Remove offensive content**

### Time Estimation

We have 200M data items to train on.

**Assumptions**

- Time to preprocess a data item = 1 ms
- Time to convert a data item to a vector = 1ms
- Time to store a vectorized data item in the database = 0.1ms

Multiply each by 200M.

1ms * 200M = 200K seconds

200K + 200K + 20K = 420K seconds
                  = 7K minutes ≈ 117 hours

We can parallelise the data processing.

16 machines can process the data in 117/16 = 7.3 hours.

**Assumption:** we are using NVIDIA A100 80GB GPUs, which provide 156 TFLOPS for TF32 calculations.

## Model Evaluation

- Perplexity
- BLEU (translation)
- ROUGE (recall)
- Track loss on the validation set

Steady decrease in perplexity and loss is good. Stagnation or increase indicates overfitting.

Further fine-tuning can raise accuracy or prep a model for a specific use case.

## Database

Use a vector DB.

## Deployment

### Storage estimation

At FP16 floating-point precision:
`Model size = 3B params * 2 bytes = 6 GB`

User profile data (assume 10KB):
`User data = 100 M * 10KB = 2TB`

User interaction data (assume 10 daily interactions at 2KB of space per interaction):
`User interaction data = 100M * 10 * 2KB = 2TB`

Indexing storage (assume average storage increase of 25% to index user data for fast retrieval):
`User interaction indexing storage = 2TB * 0.25 = 0.5TB`

Total daily storage required:
`Storage required per day = 2TB + 0.5TB = 2.5 TB/day`

Total monthly storage required:
`Monthly storage required = 2.5 TB * 30 = 75 TB/month`

### Inference estimation

Total Requests per second (TRPS):
`TRPS = (DAU * requests per day per user) / 86400 ≈ 2300`

Rate of Operations (ROps):
Rate of ops for an NVIDIA A100 for FP16 is 312 TFLOPS, and for FP32 it's 19.5 TFLOPS.

Inference Time:
`Inference Time = (Number of params * 2 * iterations/tokens per request) / Rops`
`Inference Time = (3B * 2 * 500) / 312 TFLOPS = 9.6ms`

Queries per second (QPS):
`QPS = 1/(inference time) = 1/9.6ms = 104`

Inference servers required:
`Inference servers required = TRPS / QPS = 23K / 104 ≈ 220 GPU servers`

### Bandwidth estimation

Ingress bandwidth:
Multiply TRPS by 2 to get MBps, then * 8 to get Mbps.
`Ingress bandwidth = TRPS * 2  = 23K * 2KB = 46MBps * 8 = 370Mbps`

Egress bandwidth:
Multiply TRPS by 10 to get MBps, then * 8 to get Mbps.
`Egress bandwidth = TRPS * 10 = 23K * 10 = 230MBps * 8 = 1800Mbps` 

## High-level design

- **Prompt processing:** transforms user prompts into embeddings
- **Long-term memory:** stores user interactions, feedback, and preferences. Refines input prompts 
- **Content moderation service:** scans responses for inappropriate or sensitive content
- **Data privacy service:** control panel/dashboard for compliance 
- **Knowledge base (optional):** a graph DB to capture structured relationships, concepts, and facts that the system may need to retain
- **Load Balancing:** for the embedding services and model host servers in particular

## Security

- Encryption for data in transit and at rest
- Access control measures (RBAC, OAuth)
- Audits
- Logging
- Anomaly detection

Prompt Injection Prevention:

- input sanitization
- adversarial testing
- fine-tuning models to resist manipulation















