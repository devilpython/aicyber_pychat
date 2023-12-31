import logging
from typing import Any, Dict, List, Optional

import requests

from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
)
from langchain.llms import TextGen
import json

logger = logging.getLogger(__name__)

class CUTextGen(TextGen):
    # 是否QA问答语料
    is_qa: bool = False

    presence_penalty: float = 0.0

    frequency_penalty: float = 0.0

    @property
    def _default_params(self) -> Dict[str, Any]:
        params = super()._default_params
        params['presence_penalty'] = self.presence_penalty
        params['frequency_penalty'] = self.frequency_penalty
        return params

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the textgen web API and return the output.

        Args:
            prompt: The prompt to use for generation.
            stop: A list of strings to stop generation when encountered.

        Returns:
            The generated text.

        Example:
            .. code-block:: python

                from langchain.llms import TextGen
                llm = TextGen(model_url="http://localhost:5000")
                llm("Write a story about llamas.")
        """
        if self.is_qa:
            prompt = prompt.replace('Context information is below.\n---------------------\n', 'Context information is below.\n---------------------\nCommon sense questions and answers\n')
        # print('...............prompt:')
        # print(prompt)
        if self.streaming:
            combined_text_output = ""
            for chunk in self._stream(
                prompt=prompt, stop=stop, run_manager=run_manager, **kwargs
            ):
                combined_text_output += chunk.text
            result = combined_text_output

        else:
            url = f"{self.model_url}/api/v1/generate"
            params = self._get_parameters(stop)
            # print('............params:', params)
            request = params.copy()
            request["prompt"] = prompt
            # print('...............prompt:', prompt)
            response = requests.post(url, json=request)

            if response.status_code == 200:
                result = response.json()["results"][0]["text"]
            else:
                print(f"ERROR: response: {response}")
                result = ""
            # print('...............result:', result)

        return self.__to_json(prompt, result)

    def __to_json(self, prompt, result):
        result = str(result).split("\n")[0]
        # result = str(result).split("----")[0]
        # result = str(result).split("Query:")[0]
        result = result.strip()
        if result.endswith('给出详细回答。'):
            result = result[0: len(result) - 7]
        json_data = {'from_knowledge_base': True, 'prompt': prompt, 'result': result}
        return json.dumps(json_data)
