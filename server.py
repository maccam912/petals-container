from transformers import AutoTokenizer
import torch
from petals import AutoDistributedModelForCausalLM
from typing import List, Literal
from litestar import Litestar, Controller, WebSocket, post, websocket
from pydantic import BaseModel
import logging


chat_model_name = "meta-llama/Llama-2-70b-chat-hf"
chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_name)
chat_model = AutoDistributedModelForCausalLM.from_pretrained(chat_model_name, torch_dtype=torch.float32)

model_name = "meta-llama/Llama-2-70b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoDistributedModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

logger = logging.getLogger(__name__)


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class CompletionsRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int
    temperature: float


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Content(BaseModel):
    content: str


class Role(BaseModel):
    role: str


class CompletionsChoice(BaseModel):
    text: str
    index: int
    logprobs: None
    finish_reason: str


class ChatChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Literal["length", "stop", "restart"]


class ChatDeltaChoice(BaseModel):
    index: int
    finish_reason: None | Literal["length", "stop", "restart"]
    delta: Role | Content


class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    choices: List[ChatChoice]
    usage: Usage


class CompletionsResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Content]
    usage: Usage


class ChatDelta(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatDeltaChoice]


def create_chat_prompt(messages: List[Message]) -> str:
    prompt = "[INST]"
    for m in messages:
        if m.role == "user":
            prompt += "\n" + m.content
        elif m.role == "assistant":
            prompt += "[/INST]" + m.content + "\n[INST]"
        elif m.role == "system":
            prompt += " <<SYS>>\n" + m.content + "\n<</SYS>>\n"
    prompt += "[/INST]"
    return prompt


def create_chat_response(result: str) -> ChatResponse:
    response = ChatResponse(
        id="",
        object="",
        created=0,
        choices=[
            ChatChoice(
                index=0,
                message=Message(role="assistant", content=result),
                finish_reason="length",
            )
        ],
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )
    return response

def create_response(result: str) -> CompletionsResponse:
    response = CompletionsResponse(
        id="",
        object="",
        created=0,
        model="",
        choices=[Content(content=result)],
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )
    return response

class ChatPetalsController(Controller):
    path = "/v1/chat/completions"

    @post()
    async def run(self, data: ChatRequest) -> ChatResponse:
        logger.info("In chat run")
        logger.info("Got request")
        prompt = create_chat_prompt(data.messages)
        logger.info("created prompt: " + prompt) 
        inputs = chat_tokenizer(prompt, return_tensors="pt")["input_ids"]
        logger.info("tokenized prompt, generating response...")
        outputs = chat_model.generate(inputs, max_new_tokens=512)
        logger.info("Generated response")
        resp_str = chat_tokenizer.decode(outputs[0])
        logger.info("Decoded tokens: " + resp_str)
        resp_str = resp_str.replace(prompt, "")
        resp = create_chat_response(resp_str)
        logger.info("Resp: " + str(resp))
        return resp

class PetalsController(Controller):
    path = "/v1/completions"

    @post()
    async def run(self, data: CompletionsRequest) -> ChatResponse:
        logger.info("In completions run")
        logger.info("Got request")
        prompt = data.prompt
        inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]
        logger.info("tokenized prompt, generating response...")
        outputs = model.generate(inputs, max_new_tokens=512)
        logger.info("Generated response")
        resp_str = tokenizer.decode(outputs[0])
        resp = create_response(resp_str)
        logger.info("Resp: " + str(resp))
        return resp

app = Litestar(route_handlers=[ChatPetalsController, PetalsController])