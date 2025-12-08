# Orchestration Examples

## SequentialAgent

### Job Search Example

```python
# Flow: Resume Parser → Level Classifier → A2A Deliberation → Consensus → BATCHED Job Scouts → URL Validator → Formatter

full_pipeline = SequentialAgent(
    name="job_search_orchestrator",
    sub_agents=[
        resume_parser_agent,         # Agent 1: Parse resume
        level_classifier_agent,      # Agent 2: Initial classification
        deliberation_agents,         # Agents 3-4: A2A parallel deliberation
        consensus_agent,             # Synthesize deliberation
        batched_job_scouts,          # Agent 5: BATCHED job scouts (2+2 parallel, rate-limit safe)
        url_validator_agent,         # Validate URLs from all scouts
        formatter_agent              # Agent 6: Format output
    ]
)

# Refinement pipeline (separate instances)
formatter_agent_refinement = Agent(
    name="formatter_refinement",
    model=MODEL_LITE,
    generate_content_config=RETRY_CONFIG,
    instruction=FORMATTER_INSTRUCTION,
    output_key="formatted_response"
)

refinement_pipeline = SequentialAgent(
    name="refinement_orchestrator",
    sub_agents=[
        batched_job_scouts_refinement,   # Batched scouts (2+2)
        url_validator_agent_ref,         # Validate URLs
        formatter_agent_refinement       # Format output
    ]
)

print("✅ Orchestrators defined with BATCHED parallel job scouts")
print("   - full_pipeline: 2+2 concurrent searches (~2x faster, rate-limit safe)")
print("   - refinement_pipeline: Same batched architecture")
print("")
print("   ⚡ Performance (free tier safe):")
print("      Before: ~3 min (4 sequential)")
print("      After:  ~2 min (2+2 batched parallel)")
```











