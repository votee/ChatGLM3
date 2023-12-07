# coding=utf-8
# Implements API for ChatGLM3-6B in OpenAI's format. (https://platform.openai.com/docs/api-reference/chat)
# Usage: python openai_api.py
# Visit http://localhost:8000/docs for documents.

# 在OpenAI的API中，max_tokens 等价于 HuggingFace 的 max_new_tokens 而不是 max_length，。
# 例如，对于6b模型，设置max_tokens = 8192，则会报错，因为扣除历史记录和提示词后，模型不能输出那么多的tokens。

import os
import time
import json
from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from typing_extensions import Annotated
from fastapi.security import APIKeyHeader
from openai_api_demo.schema import EmbeddingUsage, EmbeddingData, EmbeddingResponse

from utils import process_response, generate_chatglm3, generate_stream_chatglm3

MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

api_key_header = APIKeyHeader(name="Authorization")

@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan, debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class FunctionCallResponse(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "function"]
    content: str = None
    name: Optional[str] = None
    function_call: Optional[FunctionCallResponse] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None
    function_call: Optional[FunctionCallResponse] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    functions: Optional[Union[dict, List[dict]]] = None
    # Additional parameters
    repetition_penalty: Optional[float] = 1.1


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "function_call"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "function_call"]]


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(id="chatglm3-6b")
    return ModelList(data=[model_card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")

    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        repetition_penalty=request.repetition_penalty,
        functions=request.functions,
    )

    logger.debug(f"==== request ====\n{gen_params}")

    if request.stream:
        generate = predict(request.model, gen_params)
        return EventSourceResponse(generate, media_type="text/event-stream")

    response = generate_chatglm3(model, tokenizer, gen_params)

    # Remove the first newline character
    if response["text"].startswith("\n"):
        response["text"] = response["text"][1:]
    response["text"] = response["text"].strip()
    usage = UsageInfo()
    function_call, finish_reason = None, "stop"
    if request.functions:
        try:
            function_call = process_response(response["text"], use_tool=True)
        except:
            logger.warning("Failed to parse tool call, maybe the response is not a tool call or have been answered.")

    if isinstance(function_call, dict):
        finish_reason = "function_call"
        function_call = FunctionCallResponse(**function_call)

    message = ChatMessage(
        role="assistant",
        content=response["text"],
        function_call=function_call if isinstance(function_call, FunctionCallResponse) else None,
    )

    logger.debug(f"==== message ====\n{message}")

    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=message,
        finish_reason=finish_reason,
    )
    temp = response["usage"]
    logger.debug(f"==== usage ====\n{temp}")
    # return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion", usage=UsageInfo(prompt_tokens=1, total_tokens=1, completion_tokens=1))
    # task_usage = UsageInfo.model_validate(response["usage"])
    task_usage = UsageInfo(prompt_tokens=response["usage"]["prompt_tokens"], total_tokens=response["usage"]["total_tokens"], completion_tokens=response["usage"]["completion_tokens"])
    for usage_key, usage_value in task_usage.model_dump().items():
        setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)
    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion", usage=usage)


async def predict(model_id: str, params: dict):
    global model, tokenizer

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    # yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield "{}".format(chunk.json(exclude_unset=True))

    previous_text = ""
    for new_response in generate_stream_chatglm3(model, tokenizer, params):
        decoded_unicode = new_response["text"]
        delta_text = decoded_unicode[len(previous_text):]
        previous_text = decoded_unicode

        finish_reason = new_response["finish_reason"]
        if len(delta_text) == 0 and finish_reason != "function_call":
            continue

        function_call = None
        if finish_reason == "function_call":
            try:
                function_call = process_response(decoded_unicode, use_tool=True)
            except:
                logger.warning("Failed to parse tool call, maybe the response is not a tool call or have been answered.")

        if isinstance(function_call, dict):
            function_call = FunctionCallResponse(**function_call)

        delta = DeltaMessage(
            content=delta_text,
            role="assistant",
            function_call=function_call if isinstance(function_call, FunctionCallResponse) else None,
        )

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=delta,
            finish_reason=finish_reason
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        # yield "{}".format(chunk.model_dump_json(exclude_unset=True))
        yield "{}".format(chunk.json(exclude_unset=True))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    # yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield "{}".format(chunk.json(exclude_unset=True))
    yield '[DONE]'
    
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_glm_embedding(text, device="cuda"):
    global model_embedding, tokenizer_embedding
    
    # inputs = tokenizer([text], return_tensors="pt").to(device)
    # print(f"text:{text}", flush=True)
    logger.debug(f"==== embedding text ====\n{text}")
    encoded_input = tokenizer_embedding(text, padding=True, truncation=True, return_tensors="pt").to(device)
    # resp = model.transformer(**inputs, output_hidden_states=True)
    # y = resp.last_hidden_state
    # y_mean = torch.mean(y, dim=0, keepdim=True)
    # result = y_mean.cpu().detach().numpy()
    # return result
    with torch.no_grad():
        model_output = model_embedding(**encoded_input)
        # Perform pooling. In this case, mean pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    print("Sentence embeddings:", flush=True)
    print(sentence_embeddings, flush=True)
    return sentence_embeddings
  
  
@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    # _: Annotated[str, Depends(api_key_header)], 
    # text: Annotated[str, Body(embed=True)] = None
    model: str | None = None,
    input: List[str] = Body(..., embed=True)
) -> EmbeddingResponse:
    embedding_obj = get_glm_embedding(input)
    embedding_list = embedding_obj.tolist()
    output_item = []
    for index, embedding_item in enumerate(embedding_list):
        # output_item.append({
        #     "embedding": embedding_item,
        #     "index": index,
        #     "object": "embedding"
        # })
        output_item.append(EmbeddingData(embedding=embedding_item, index=index, object="embedding"))
    return EmbeddingResponse(data=output_item, model="text-embedding-ada-002", object="list", usage=EmbeddingUsage(prompt_tokens=1, total_tokens=1))
    # return_dict = {"data": output_item, "model": "text-embedding-ada-002", "object": "list", "usage": {"prompt_tokens": 1, "total_tokens": 1}}
    # json_dict = json.dumps(return_dict)
    # # print(f"create_embeddings json_str={json_str}", flush=True)
    # return json_dict


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    if 'cuda' in DEVICE:  # AMD, NVIDIA GPU can use Half Precision
        model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE).eval()
    else:  # CPU, Intel GPU and other GPU can use Float16 Precision Only
        model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).float().to(DEVICE).eval()
    tokenizer_embedding = BertTokenizer.from_pretrained("./text2vec-large-chinese/vocab.txt", trust_remote_code=True,local_files_only=True)
    model_embedding = BertModel.from_pretrained("./text2vec-large-chinese/pytorch_model.bin",config='./text2vec-large-chinese/config.json', trust_remote_code=True, local_files_only=True).cuda()
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
