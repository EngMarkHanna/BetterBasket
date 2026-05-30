# LLM Smoke Test

Model deployment: `gpt-5.4-nano`
Endpoint: `https://betterbasketopenai.openai.azure.com/openai/v1/`

## 1. Connectivity
- Latency: 2.286s
- finish_reason: `stop`
- content: `'OK'`
- usage: `{'prompt_tokens': 14, 'completion_tokens': 5, 'total_tokens': 19, 'reasoning_tokens': 0, 'accepted_prediction_tokens': 0, 'cached_prompt_tokens': 0}`

## 2/3. Single-pair structured output (effort=default)
- n: 10
- latency p50/p90: 0.989s / 1.168s
- JSON validity: 100%
- Label agreement (vs my expected): 90%
- Avg tokens per pair (prompt / completion): 359.5 / 45.4
- Total reasoning tokens across n pairs: 0

## 2/3. Single-pair structured output (effort=minimal)
- n: 10
- latency p50/p90: 0.839s / 1.036s
- JSON validity: 100%
- Label agreement (vs my expected): 90%
- Avg tokens per pair (prompt / completion): 359.5 / 44.9
- Total reasoning tokens across n pairs: 0

## 4. Per-prompt candidate batching
- K (candidates per prompt): 5
- Latency: 1.46s
- Usage: {'prompt_tokens': 496, 'completion_tokens': 88, 'total_tokens': 584, 'reasoning_tokens': 0, 'accepted_prediction_tokens': 0, 'cached_prompt_tokens': 0}
- Parsed: `{'best_candidate_index': 0, 'is_match': True, 'match_type': 'exact_national_brand', 'confidence': 0.98, 'reason_codes': ['Same national brand (Russell Stover).', 'Same product type: Sugar Free Assorted Christmas Candy Gift Box.', 'Compatible size: 6.2 oz vs 6.2 ounce.', 'Same seasonal/holiday candy category and gift box format.']}`
- Expected best index: 0

## 5. Batch API submission
- Submission FAILED: `Error code: 400 - {'errors': {'data': [{'code': 'invalid_deployment_type', 'message': "The current deployment 'gpt-5.4-nano' is using the SKU 'GlobalStandard', which is not supported for batch jobs. Please create a new deployment using one of the supported SKUs: [globalbatch, datazonebatch], and then retry the operation.", 'line': None, 'param': 'model'}], 'object': 'list'}}`