# Ko-Gemma
  [![Huggingface](https://img.shields.io/badge/Huggingface-ko--gemma--2b--v1-%23800020?style=flat&logo=Pytorch&logoColor=white)](https://huggingface.co/nlpai-lab/ko-gemma-2b-v1)
  [![Huggingface](https://img.shields.io/badge/Huggingface-ko--gemma--7b--v1-%23800020?style=flat&logo=Pytorch&logoColor=white)](https://huggingface.co/nlpai-lab/ko-gemma-7b-v1)
  
<div id="top" align="center">

   <img src="https://github.com/KU-HIAI/Ko-Gemma/assets/60927808/e217e02b-2a52-42d7-bb9a-eab7b1739696" height="300" alt="logo">

**Ko-Gemma**: Korean Gemma

| [한국어](README.md) | [English](docs/README-en.md) |

</div>

## What's New
- **February 2024: [🚀 Model Release 🚀]** We are excited to announce the release of our initial models for the Korean language processing community! Check them out: 
  - [ko-gemma-2b-v1](https://huggingface.co/nlpai-lab/ko-gemma-2b-v1)
  - [ko-gemma-7b-v1](https://huggingface.co/nlpai-lab/ko-gemma-7b-v1)


## Ko-Gemma

- Gemma는 Gemini 모델을 만들기 위해 사용된 동일한 연구와 기술로 구축된 경량화된 최신 대규모 언어모델입니다.
- Ko-Gemma는 Google의 [Gemma](https://blog.google/technology/developers/gemma-open-models/)에 의해 개발된 한국어 대규모 언어 모델로, 고려대학교 [NLP & AI LAB](https://github.com/nlpai-lab)와 [HIAI 연구소](http://hiai.kr/intro/intro-vision/)가 협력하여 KULLM 데이터셋 및 수제작한 데이터셋을 활용하여 한국어 성능을 더욱 향상시킨 버전입니다.
- Ko-Gemma v1은 `chat_template`만을 사용하여 학습되었으므로, `tokenizer.chat_template`을 사용하지 않는 생성 또는 평가 작업에서는 성능이 크게 낮아질 수 있습니다.
- 앞으로 우리는 한국어 데이터로 사후 학습을 거친 모델, 다양한 한국어 지시어 데이터셋으로 튜닝된 모델, 한국어에 더 최적화된 전이 모델을 출시할 계획입니다.


## Quick start

```python
from transformers import AutoTokenizer, pipeline
import torch

model_and_tokenizer_path = "nlpai-lab/ko-gemma-7b-v1"

tokenizer = AutoTokenizer.from_pretrained(model_and_tokenizer_path)
pipeline = pipeline(
    "text-generation",
    model=model_and_tokenizer_path,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

tokenizer = AutoTokenizer.from_pretrained(model_and_tokenizer_path)
messages = [
    {"role": "user", 
     "content": "이순신 장군에 대해 설명해주세요."},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # Don't use `pipeline.tokenizer`
print(prompt) # <bos><start_of_turn>user\n이순신 장군에 대해 설명해주세요.<end_of_turn>\n<start_of_turn>model\n

outputs = pipeline(
    prompt,
    max_new_tokens=4096,
    add_special_tokens=True
)
print(outputs[0]["generated_text"][len(prompt):]) # '이순신 장군은 조선 시대의 대표적인 군사 지도자이자 전략가입니다. 그는 조선 시대의 수도인 한양에서 태어났으며, 조선 시대 군대에서 다양한 지도자로 활동했습니다.\n\n이순신 장군의 가장 주목할 만한 업적 중 하나는 1592년부터 1598년까지 일본이 조선을 침공한 일본 전쟁에서의 활동입니다. 이 전쟁에서 이순신 장군은 조선 군대의 전략적인 지도자로 활동하며 일본의 침략에 저항하는 데 큰 역할을 했습니다.\n\n이순신 장군은 전투에서의 용기와 전술적 지성으로 유명했습니다. 그는 전투에서 전술적인 사고를 발휘하고 적의 약점을 공격하는 것으로 유명했습니다. 또한 그는 조선 군대의 전력을 고취하고 전투에서 승리하는 데 도움이 되는 연설과 격려의 말을 전하는 것으로도 유명했습니다.\n\n이순신 장군은 전쟁이 끝난 후에도 조선 군대에서 계속 활동하며 조선 군대의 지도자로 활동했습니다. 그는 조선 군대의 전력을 유지하고 조선의 안보를 지키는 데 큰 역할을 했습니다.\n\n이순신 장군은 조선 시대의 대표적인 군사 지도자이자 전략가로 기억되고 있습니다. 그의 용기와 전술적 지성, 그리고 조선 군대의 전력을 유지하는 데 기여한 공로는 그를 전설적인 인물로 만들었습니다.'
```


## KoBEST에 대한 LLM 추론 결과

한국어 균형 평가 작업([[Korean Balanced Evaluation of Significant Task](https://arxiv.org/abs/2204.04541)], M.Jang et. al)에 대해 5-shot으로 평가된 결과입니다.

| Model                          | WIC Acc | WIC F1 | Sentineg Acc | Sentineg F1 | Hellaswag Acc | Hellaswag F1 | COPA Acc | BoolQ Acc | BoolQ F1 | Avg F1 | Avg F1 (Excl. Sentineg, WIC) |
|--------------------------------|---------|--------|--------------|-------------|---------------|--------------|----------|-----------|----------|--------|-----------------------------|
| google/gemma-2b-it             | 0.528   | 0.520  | 0.655        | 0.631       | 0.35          | 0.347        | 0.42     | 0.526     | 0.525    | 0.536  | 0.510                       |
| google/gemma-7b-it             | 0.508   | 0.506  | 0.736        | 0.722       | 0.352         | 0.349        | 0.42     | 0.544     | 0.543    | 0.573  | 0.545                       |
| nlpai-lab/ko-gemma-2b-v1       | 0.521   | 0.420  | 0.587        | 0.517       | 0.354         | 0.350        | 0.456    | 0.595     | 0.594    | 0.506  | 0.532                       |
| nlpai-lab/ko-gemma-7b-v1       | 0.563   | 0.542  | 0.788        | 0.780       | 0.39          | 0.388        | 0.488    | 0.639     | 0.638    | 0.629  | 0.608                       |

- 평가는 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/)를 사용하였습니다. [add_bos_token 옵션](https://github.com/EleutherAI/lm-evaluation-harness/pull/1465)을 키고 평가하였습니다.

## License
License
Copyright 2024 DeepMind Technologies Limited

This code is licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
