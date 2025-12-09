# Envelope Estimates

## Core Questions

1. How many calculations will be performed in total?
2. How many calculations can our computer(s) do in one second?

## FLOPs (Floating-Point Operation)

When we talk about training an AI, we are really talking about performing billions or trillions of these simple math operations.

## Training Time Formula

Training Time (in seconds) = (Total FLOPS required) / (FLOPS our computer can do per second)

### Deconstructing the Formula

These factors impact the total # of FLOPs required:

- **Number of Parameters:** # of internal weights updated during training. This reflects the model's size (GPT has billions of parameters).
- **FLOPS per data item:** # of calculations needed to process a single piece of data (one forward and backward pass)
- **Epochs:** # of times the model will see the entire dataset during training
- **Data Items:** total # of items in the dataset, like sentences or images
- **Model Architecture Constant:** A constant representing model-specific workload, such as the number of generation steps (e.g., for a Diffusion model) or iterations.
- **Rate of Operations:** speed of a single GPU, measured in FLOPS per second
- **Total Number of GPUs**

Ultimately, the # of FLOPs required is the total number of math problems you have to solve.

For an estimate, the # of FLOPs per data item is `6 * N_params`. That's because:

• A feedforward pass (making a prediction) takes about 2 * N_params calculations.
• A backward pass (learning from the prediction) takes about 4 * N_params calculations.

---
Calculating the FLOPs we can do per second:

It’s the speed of one computer multiplied by the number of computers you have working together.

## Inference Time Formula

`Time to process one inference = (num_params * 2 * C) / RateOfOperations`

where C is the model architecture constant (see above) and RateOfOperations is the speed of one GPU, measured in FLOPs/s.

## Model Deployment 

Assumption: we are serving 100 million daily active users (DAU).

Key questions:

- Storage: How much disk space do you need?
- Inference Servers: How much computing power is required?
- Network Bandwidth: How much data will be flowing in and out?

### Storage Requirements

Disk space required to store the LLM itself.

**Formula:** 

`Model storage = No. of parameters × Data precision`

**Data precision:**

Size of a single parameter (in bytes). Options include:

◦ FP64 (double precision): 8 bytes per parameter.
◦ FP32 (full precision): 4 bytes per parameter.
◦ FP16 (half precision): 2 bytes per parameter.
◦ Quantized (8-bit): 1 byte per parameter.

Higher precision (like FP32) means greater accuracy but results in a larger model file.

Lower precision (like FP16) can significantly reduce the model's size and improve performance, which helps optimize costs.

**Example Calculation**

Storage for a 3 billion parameter model using FP16 precision.

Model storage = 3 Billion parameters × 2 Bytes/parameter
              = 6 GB

**Real model examples**

Llama 3.2 (3B params) * FP16 (x2) = 6GB
Llama 3.2 (3B params) * FP32 (x4) = 12GB
Stable Diffusion 3.5 Large (8.1B) * FP16 (x2) = 16.2GB
Stable Diffusion 3.5 Large (8.1B) * FP64 (x8) = 64.82GB

### User Profile Data Storage

Storage needed for all user metadata, such as account information, preferences, and other profile details.

**Formula**

`User data = Number of users × Storage required per user`

**Example Calculation** 

For a system with 100 million users, where each user's profile takes about 10 KB:

User data = 100M users × 10 KB/user
          = 1 TB

### User Interaction Data Storage

Storage needed each day to log user interactions with the model, such as prompts and responses.

**Formula**

`User interaction data = Number of users × Daily interactions × Single interaction size`
                       = DAU * requests/day * storage space for one prompt/response pair

We could assume 2KB per prompt/response pair.

User interaction data = 100M users × 10 interactions/day × 2 KB/interaction
                      = 2 TB per day

### Total Daily Storage

We must also account for indexing.

Indexing creates special data structures that help the system retrieve user interaction data quickly, but this requires additional storage.

**Formula**

`Storage required per day = Users interaction data + Indexing storage`

Indexing storage is typically estimated as a percentage of the data it's indexing. Let's assume it requires an additional 25% of the user interaction data storage.

**Example Calculation**

Indexing storage = 2 TB × 0.25
                 = 0.5 TB per day

Total daily storage = 2 TB + 0.5 TB 
                    = 2.5 TB per day

Monthly storage = 2.5 TB/day × 30 days = 75 TB/month

Note that real production systems also need redundancy, so actual systems will have higher storage requirements than this.

## Inference Server (GPU) Requirements

How many GPUs are needed to handle the expected volume of user requests without causing delays?

We can break this down in 4 steps.

### Total Requests Per Second (TRPS)

`TRPS = (No. of users × Requests per day per user) / 86400 seconds`

**Example**

For 100 million users making 10 requests per day:

TRPS = (100,000,000 users × 10 requests/day) / 86,400 seconds
     ≈ 11,574 requests per second

Note: this calculation assumes a uniform distribution of requests throughout the day.


### Inference Time per Request

How long does it take for a single GPU to process one request?

**Formula**

`T_inference = (N_params × 2 × C) / R_ops`

Where R_ops is rate of operations for the GPU, measured in FLOPS, and C is the # of iterations (e.g. tokens). 

For an NVIDIA A100 GPU, the rates are:

- FP16: 312 TFLOPS (Trillion FLOPS)
- FP32: 19.5 TFLOPS

**Example Calculation**

Time to generate 500 tokens with a 3B parameter model on an NVIDIA A100 GPU using FP16 precision.

T_inference = (3B parameters × 2 × 500 tokens) / 312 TFLOPS
            = 30B / 312
            = 9.6 milliseconds (ms)

Note: In real-world scenarios, other factors like communication overhead, data transfer, pre-processing inputs, and post-processing outputs also contribute to the overall time a user waits for a result.

### Queries Per Second per GPU (QPS)

How many requests one GPU can handle per second.

It's simply the inverse of the inference time.

**Formula**

`QPS = 1 / T_inference`

**Example Calculation** 

Using the inference time from the previous section:

QPS = 1 / 9.6 ms
    = 1 / 0.0096 seconds
    ≈ 104 QPS

This means a single NVIDIA A100 GPU can process approximately 104 of these specific requests every second.

### Total GPUs Required

Divide the total requests our system needs to handle (TRPS) by the number of requests a single GPU can handle (QPS) to find the total number of GPUs needed.

**Formula**

Inference servers required = TRPS / QPS

**Example Calculation**

Inference servers required = 11,574 TRPS / 104 QPS
                           ≈ 112 GPUs


Note: techniques like model quantization, batching, and sharding can reduce GPU requirement significantly, sometimes by up to 50%.

## Network Bandwidth

How much data can be sent and received per second.

### Ingress Bandwidth

Capacity needed to handle all incoming user requests.

We first need to assume an average request size.

A typical assumption is 2 KB per request, which accounts for metadata, headers, and the user's input prompt.

**Formula**

`Ingress bandwidth = TRPS × Request size`

**Example Calculation**

Using our TRPS of 11,574 and a 2 KB request size:

Ingress bandwidth = 11,574 requests/sec × 2 KB/request
                  = 23,148 KBps
                  ≈ 23.15 MBps

To convert MegaBytes per second (MBps) to Megabits per second (Mbps), multiply by 8.

Ingress bandwidth = 23.15 MBps × 8
                  ≈ 186 Mbps

### Egress Bandwidth

Capacity needed to send the model's generated responses back to all users.

For a text-to-text generation system, a reasonable assumption is an average response size of 10 KB (about 1K characters).

**Formula**

`Egress bandwidth = TRPS × Response size`

**Example Calculation**

Using our TRPS of 11,574 and a 10 KB response size:

Egress bandwidth = 11,574 requests/sec × 10 KB/request
                 = 115,740 KBps
                 ≈ 115.74 MBps

Conversion to Mbps:

Egress bandwidth = 115.74 MBps × 8
                 ≈ 926 Mbps

Key Insight: A simple text response (like our 10 KB example) will require far less bandwidth than a generated image or video, which could be several megabytes (MBs) in size, dramatically increasing the required egress bandwidth.








