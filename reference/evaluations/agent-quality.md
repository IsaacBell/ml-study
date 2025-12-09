# Agentic AI Quality

Traditional software verification asks, "Did we build the product right?". Modern AI asks, "Did we build the right product?". 

In traditional ML/deep learning systems, statistical metrics (like F1 score, recall) are used to evaluate regression/classification models.

Agentic AI systems require a new paradigm, as their technical shifts render old practices outmoded. The primary focus of eval is no longer the model, but the **system trajectory**.

## Agent Failures

- Bias
- Hallucination/factual errors
- Performance/concept drift
- (Unintented) emergent behaviors

Root cause analysis requires data analysis, model retraining, and system evals.

Multi-agent systems (MAS) introduce further complexity:

- **Emergent System Failures**: resource contention, communication bottlenecks, system deadlocks
- **Cooperative Eval**: tracking success as a global metric (e.g. supply chain)
- **Competitive Eval**: tracking agent performances individually, plus stability of market/environment (e.g. auction systems)

## Strategic Shift: Outside-In Evaluation

### Four Pillars

1. Effectiveness (goal achievement)
2. Efficiency (operational cost)
3. Robustness (reliability)
4. Safety/Alignment (trustworthiness)

**Effectiveness:** did the agent achieve the user's real intent? Did it drive KPIs/business metrics? Did it drive conversion, produce the right insight, etc.?

**Efficiency:** Did the agent solve the problem _well_? Did it minimise token cost, wall-clock time (latency), and trajectory complexity (# of steps/loops taken)?

**Robustness:** How does the agent handle adversity? API timeouts, website layout updates, ambiguous prompts, failure scarions, etc. Does it retry failed tasks, get user clarification, and report what went wrong and why (rather than hallucinate/crash)?

**Safety/Alignment:** Does the agent stay within ethical bounds? Is it fair and non-biased? Is it secure against prompt injection attacks and data leakage possibilities? Does the agent stay on task, operate as expected as a representative of your company, and refuse harmful instructions?

## Judgment and Validation

### End-to-End Eval

- Layers of Eval
    - Output
        - task success rate
        - user satisfaction
        - quality
    - Process
        - planning
        - tool use
        - memory
- Judgment Methods
    - Automated Metrics
    - LLM-as-Judge
    - Agent-as-Judge
    - Human In the Loop (HITL)
    - User Feedback/Reviewer UI
- Safety Eval
    - Fairness
    - Bias
    - Truthfulness
    - Privacy
    - Compliance

Starting from the outside in, we might start by defining core objectives. Metrics, at this point, focus on task completion.

- Success Rate: a binary or graded score. e.g. PR acceptance rate, CX issue resolution rate
- User Satisfaction: direct feedback scores (CSAT, likes/dislikes)
- Quality: could be quantitative or qualitative. Quantitative examples: accuracy, completeness (were all 5 articles summarized?)

As failures are identified, we can look inside the to assess components and internals. Here we inspect **the process:**

- Planning
    - Core reasoning: is the LLM at fault? Check for hallucinations, off-topic responses, context pollution, and repetitive loops.
    - Tool selection & parameterization: are the right tools being called, with real (not hallucinated) parameters, types, and data?
    - Tool response interpretation: does the LLM make the right takeaways/observations from data retrieved from tools? Does it misinterpret numerical data, failing to parse structured data properly, or treat errors as if they were successful?
    - RAG performance: are relevant, up-to-date documents retrieved, and does the LLM actually consider (not disregard) the info retrieved?
    - Trajectory efficiency: is there inefficient resource allocation (high # of API calls, high latency, redundancy)?
    - MAS Dynamics: any issues with communication loops, inter-agent misunderstandings, or agents drifting from their defined role?

### Judgment Approaches

Hybrid judgment is the most common method of review.

**Automated Metrics**

Efficient but shallow. Generally the first quality gate in CI/CD. Great as a "first filter" for quality. 

Use them for regression testing.

- String similarity (BLEU, ROUGE) comparing text to references
- Embedding similarity (BERTScore, cosine similarity) checking semantic closeness
- Task-specific benchmarks (TruthfulQA)

**LLM-as-Judge**

Use a (powerful) LLM to evaluate another agent's output.

Use this to score final responses.

A typical prompt includes a reference answer, the original user prompt, the agent's response, and a detailed scoring rubric ("Rate the helpfulness, correctness, and safety of this response on a scale of 1-5, explaining your reasoning.").

This is fast and scales to thousands of scenarios, enabling iterative evaluation.

**Pairwise LLM-as-Judge**

Prioritize pairwise comparison over single-scoring to mitigate bias. Run the eval prompt against two agent versions (e.g., "main" vs. "staging") and generate an "Answer A" and "Answer B" for each prompt.

Example prompt:

"Given this User Query, which response is more helpful: A or B? Explain your reasoning."

Full example:

```
You are an expert evaluator for a customer support chatbot. Your goal is to assess which of two responses is more helpful, polite, and correct.

[User Query]
"..."

[Answer A]
"..."

[Answer B]
"..."

Please evaluate which answer is better. Compare them on correctness, helpfulness, and tone. Provide your reasoning and then output your final decision in a JSON object with a "winner" key (either "A", "B", or "tie") and a "rationale" key.
```

## Agent-as-Judge

Agents require a deeper eval than just their final output.

Agent-as-Judge is an extension/evolution of the LLM-as-judge technique, using an agent to assess process, not just output.

**Eval Dimensions:**

- Plan Quality: was the plan logically structured and doable?
- Tool Use: were the right tools chosen and used well?
- Context handling: did the agent use prior info effectively?

**Steps:**

- Configure agent-under-eval to log and export traces (including internal plan, list of tools used, arguments passed)
- Create a Critic Agent that evaluates the trace

**Prompt Rubric:**

1. Based on the trace, was the initial plan logical?
2. Was the {tool_A} tool the correct first choice, or should another tool have been used?
3. Were the arguments correct and properly formatted?

The goal is to identify process failures, even if the final output is correct.

## Human-in-the-Loop (HITL) Evaluation

@TODO

**Steps:**

- Implement an [interruption workflow](https://google.github.io/adk-docs/tools/confirmation/). 
- Configure the agent to pause execution before commiting to a high-stakes action. 
- Surface the agent state and planned action(s) to a Reviewer UI, where a human will approve/reject next steps

## User Feedback and Reviewer UI

****Best Practices:****

- Low-friction feedback: thumbs up/down, quick sliders, or short comments.
- Context-rich review: feedback should be paired with the full conversation and agent’s reasoning trace.
- Two-panel reviewer interface: conversation on the left, reasoning steps on the right, with inline tagging for issues like “bad plan” or “tool misuse.”
- Governance dashboards: aggregate feedback to highlight recurring issues and risks.

A user feedback system as an event-driven pipeline, not a static log. 

When a user clicks "thumbs down," capture the full, context-rich conversation trace and add it to a dedicated review queue in the Reviewer UI.

## Guardrails

Build a reusable plugin (e.g. class SafetyPlugin) with methods like the following:

- check_input_safety(): a `before_model_callback` that runs a prompt injection classifier
- check_output_pii(): an `after_model_callback` with a PII scanner









