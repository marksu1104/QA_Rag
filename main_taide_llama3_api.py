import uvicorn
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, List
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import time
import os
import sys
import logging

# 定義 logging 輸出格式
FORMAT = '%(asctime)s %(filename)s %(levelname)s:%(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)

llms = {}
dirname = os.path.dirname(os.path.realpath(sys.argv[0]))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 指定要載入的模型，可以根據需求選擇不同的模型
    model_name = "Llama3-TAIDE-LX-8B-Chat-Alpha1"
    checkpoint = os.path.join(dirname, model_name)

    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        device_map="cuda",  # 指定使用 GPU 加速
        torch_dtype=torch.bfloat16,  # 指定模型的數據型別為 bfloat16，以減少內存使用
    )

    # 載入模型的 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

    # 指定生成文本時的終止符號（例如<|eot_id|>）
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # 建立一個文本生成的 pipeline，定義生成文本的參數
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=8000,  # 約是 8B 模型的限制，包含了模型輸入及輸出的總 token 數
        eos_token_id=terminators,  # 使用指定的終止符號
    )

    llms["taide"] = text_generation_pipeline
    yield

# 初始化 FastAPI
app = FastAPI(lifespan=lifespan)


# 執行 LLM 生成
def run_llm(messages_param) -> AsyncGenerator:
    llm: pipeline = llms["taide"]

    # 計時開始
    t1 = time.perf_counter()

    # 根據模板生成 prompt
    prompt = llm.tokenizer.apply_chat_template(
        messages_param,
        tokenize=False,
        add_generation_prompt=True
    )

    # 使用 LLM 生成文本，並進行參數設置
    '''
    [注意] 競賽每題生成最大 token length 上限為 400，超過該題不計分
    '''
    outputs = llm(
        prompt,
        max_new_tokens=400,  # 生成的最大 token 數
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    # 計時結束
    t2 = time.perf_counter()
    logging.info(f"Execution time: {t2 - t1}")

    torch.cuda.empty_cache()

    return outputs[0]["generated_text"][len(prompt):]


# 定義 POST，接收生成的回應
@app.post("/chat")
async def root(question: str = '',
               prompts: str = "你是一個來自台灣的AI助理，你的名字是 TAIDE，樂於以台灣人的立場幫助使用者，會用正體中文回答問題。",
               assistant: str = '',
               message_self: bool = False,
               message: List[dict] = []) -> StreamingResponse:
    if not message_self:
        messages_param = [
            {"role": "system", "content": prompts},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant},
        ]
    else:
        messages_param = message

    return StreamingResponse(run_llm(messages_param), media_type="text/event-stream")

# 啟動 FastAPI，指定 host & port
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8087)
