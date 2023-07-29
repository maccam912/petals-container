from transformers import AutoTokenizer
import torch
from petals import AutoDistributedModelForCausalLM
from typing import List, Literal
from litestar import Litestar, Controller, WebSocket, post, websocket
from pydantic import BaseModel
import logging


model_name = "meta-llama/Llama-2-70b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoDistributedModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

logger = logging.getLogger(__name__)


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Request(BaseModel):
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


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Literal["length", "stop", "restart"]


class DeltaChoice(BaseModel):
    index: int
    finish_reason: None | Literal["length", "stop", "restart"]
    delta: Role | Content


class Response(BaseModel):
    id: str
    object: str
    created: int
    choices: List[Choice]
    usage: Usage


class Delta(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[DeltaChoice]


def create_prompt(messages: List[Message]) -> str:
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


def create_response(result: str) -> Response:
    response = Response(
        id="",
        object="",
        created=0,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=result),
                finish_reason="length",
            )
        ],
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )
    return response

class PetalsController(Controller):
    path = "/v1/chat/completions"

    @post()
    async def run(self, data: Request) -> Response:
        logger.info("In run")
        logger.info("Got request")
        prompt = create_prompt(data.messages)
        logger.info("created prompt: " + prompt) 
        inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]
        logger.info("tokenized prompt, generating response...")
        outputs = model.generate(inputs, max_new_tokens=512)
        logger.info("Generated response")
        resp_str = tokenizer.decode(outputs[0])
        logger.info("Decoded tokens: " + resp_str)
        resp_str = resp_str.replace(prompt, "")
        resp = create_response(resp_str)
        logger.info("Resp: " + str(resp))
        return resp

@websocket("/ide/ws")
async def wss(socket: WebSocket) -> None:
    await socket.accept()
    while True:
        try:
            data = await socket.receive_text()
        except websocket.WebSocketDisconnect:
            return
        await socket.send_text(data)

app = Litestar(route_handlers=[PetalsController])
