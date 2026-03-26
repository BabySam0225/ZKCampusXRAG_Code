"""
generator.py  —  LLM   ⽣成层
⽀持多轮对话记忆 ：   history   参数传⼊历史消息列表
"""
from dataclasses import dataclass,  field 
from typing import List, Optional
from   rich.console   import   Console
from rag_config import RAGConfig,  DEFAULT_CONFIG 

console = Console()

#   ─────────────────────────────────────────────
#  Prompt   模板
#   ─────────────────────────────────────────────
SYSTEM_PRoMPT  = """ 你是⼀个严格基于知识库回答问题的助⼿ 。
核⼼规则 ：
1.  你必须仅根据提供的 【   参考上下⽂ 】   回答问题
2.  如果上下⽂中没有⾜够信息 ，   必须回答 ：   " 未在知识库中找到相关信息 "
3.  禁⽌使⽤上下⽂以外的知识
4.  回答时必须引⽤来源编号 ，   格式 ：   [ 来源   N]
5.  可以结合对话历史理解⽤户的追问意图 ，  但答案内容必须来⾃上下⽂
6.  回答要简洁、   准确、   有结构 """

ANSWER_PRoMPT  = """ 参考上下⽂ ：
{context}
⽤户问题 ：   {question}
请基于以上上下⽂回答问题 ，   并在回答中标注引⽤的来源编号 （   如   [ 来源   1] ）   。   """

#   ─────────────────────────────────────────────
#  对话历史条⽬
#   ─────────────────────────────────────────────
@dataclass
class RAGAnswer: 
    question: str 
    answer: str 
    sources:   List[dict]
    model:   str
    
def   _build_messages( 
    question: str, 
    context: str,
    history:   List[dict],   #   [{\"role\":   \"user/assistant\",   \"content\":   \"...\"},   ...]
)   ->   List[dict]:  
    """
    组装发给  LLM   的完整消息列表 ：
    system
    [ 历史   user/assistant  轮次 ]	←   让   LLM   知道之前聊了什 么
    当前   user （   含检索到的上下⽂ ）
    """
    messages = [{"role": "system", "content":  SYSTEM_PRoMPT}] 
    #  加⼊历史 （  最多保留最近  MAX_HISToRY  轮 ，  避免  token  爆炸 ）   
    MAX_HISToRY = 6  #  3 轮问答  = 6 条消息  
    messages.extend(history[-MAX_HISToRY:])
    #  当前问题 （  带上下⽂ ）
    prompt = ANSWER_PRoMPT.format(context=context,  question=question) 
    messages.append({"role": "user", "content": prompt})
    return   messages

#   ─────────────────────────────────────────────
#  DeepSeek   API   ⽣成
#   ─────────────────────────────────────────────
class   DeepSeekGenerator:
    def    __init__   (self, config:  RAGConfig): 
        import httpx
        from   openai   import   OpenAI 
        self.client = OpenAI(
            api_key=config.deepseek_api_key, 
            base_url=config.generator.deepseek_base_url, 
            http_client=httpx.Client(verify=False),
        )
        self.cfg   =   config.generator
        
    def generate(self, question: str, context: str, history: List[dict]) ->  str: 
        messages = _build_messages(question, context, history)
        response =  self.client.chat.completions.create(  
            model=self.cfg.deepseek_model, 
            messages=messages, 
            max_tokens=self.cfg.max_tokens, 
            temperature=self.cfg.temperature,
        )
        return   response.choices[0].message.content

#   ─────────────────────────────────────────────
#  Anthropic   API   ⽣成
#   ─────────────────────────────────────────────
class   AnthropicGenerator:
    def    __init__   (self, config:  RAGConfig): 
        import anthropic
        self.client =  anthropic.Anthropic(api_key=config.anthropic_api_key) 
        self.cfg = config.generator
        
    def generate(self, question: str, context: str, history: List[dict]) ->  str: 
        messages = _build_messages(question, context, history)
        #  Anthropic   的  system   要单独传 ，   不放在  messages   ⾥
        system =  messages[0]["content"] 
        user_messages = messages[1:]
        response =  self.client.messages.create(  
            model=self.cfg.anthropic_model, 
            max_tokens=self.cfg.max_tokens, 
            system=system, 
            messages=user_messages,
        )
        return   response.content[0].text

#   ─────────────────────────────────────────────
#  openAI   API   ⽣成
#   ─────────────────────────────────────────────
class   OpenAIGenerator:
    def    __init__   (self, config:  RAGConfig): 
        import httpx
        from   openai   import   OpenAI 
        self.client = OpenAI(
            api_key=config.openai_api_key, 
            http_client=httpx.Client(verify=False),
        )
        self.cfg   =   config.generator
        
    def generate(self, question: str, context: str, history: List[dict]) ->  str: 
        messages = _build_messages(question, context, history)
        response =  self.client.chat.completions.create(  
            model=self.cfg.openai_model, 
            messages=messages, 
            max_tokens=self.cfg.max_tokens, 
            temperature=self.cfg.temperature,
        )
        return   response.choices[0].message.content

#   ─────────────────────────────────────────────
#  本地模型⽣成
#   ─────────────────────────────────────────────
class   LocalGenerator:
    def    __init__   (self, config:  RAGConfig): 
        self.cfg = config.generator 
        self.tokenizer = None
        self.model   =   None
        
    def   _load(self):
        if   self.model   is   not   None:  
            return
        from transformers import AutoTokenizer, AutoModelForCausalLM,  BitsAndBytesConfig 
        import torch
        console.print(f"[cyan] 加载⽣成模型 :  {self.cfg.local_model}[/cyan]")
        self.tokenizer = AutoTokenizer.from_pretrained( 
            self.cfg.local_model,  
            trust_remote_code=True
        )
        quant_config = BitsAndBytesConfig(load_in_4bit=True,  bnb_4bit_compute_dtype=torch.float16) if self.cfg.load_in_4bit else None
        self.model =  AutoModelForCausalLM.from_pretrained(  
            self.cfg.local_model,
            device_map="auto", 
            quantization_config=quant_config, 
            trust_remote_code=True,
        )
        
    def generate(self, question: str, context: str, history: List[dict]) ->  str:  
        self._load()
        import   torch
        messages = _build_messages(question, context,  history) 
        text = self.tokenizer.apply_chat_template(
            messages,   
            tokenize=False,   
            add_generation_prompt=True
        )
        inputs = self.tokenizer(text,  return_tensors="pt").to("cuda") 
        with torch.no_grad():
            outputs   =   self.model.generate(
                **inputs, 
                max_new_tokens=self.cfg.max_tokens, 
                temperature=self.cfg.temperature,  
                do_sample=(self.cfg.temperature >  0),
            )
        return self.tokenizer.decode(   
            outputs[0][inputs["input_ids"].shape[1]:],  
            skip_special_tokens=True
        ).strip()

#   ─────────────────────────────────────────────
#  统⼀⽣成接⼝
#   ─────────────────────────────────────────────
class   Generator:
    def    __init__   (self, config: RAGConfig =  DEFAULT_CONFIG): 
        self.cfg = config.generator
        provider = self.cfg.api_provider if self.cfg.mode == "api" else  "local" 
        if provider == "deepseek":
            self._backend =  DeepSeekGenerator(config) 
        elif provider == "anthropic":
            self._backend =  AnthropicGenerator(config) 
        elif provider == "openai":
            self._backend =  OpenAIGenerator(config)  
        else:
            self._backend   =   LocalGenerator(config)
            
    def generate(  
        self,  
        question:   str, 
        context: str,
        compressed_chunks:   List[dict],
        history:   List[dict]  =  None,   # ←  新增 ，  可选
    )   ->   RAGAnswer:
        console.print(f"\\n[bold   cyan]\\nLLM   ⽣成 [/bold   cyan]...")
        answer = self._backend.generate(question, context, history or  []) 
        model_name = (
            self.cfg.deepseek_model if self.cfg.api_provider ==  "deepseek" 
            else self.cfg.local_model if self.cfg.mode == "local"
            else self.cfg.anthropic_model if self.cfg.api_provider ==  "anthropic" 
            else self.cfg.openai_model
        )
        return RAGAnswer(  
            question=question, 
            answer=answer, 
            sources=compressed_chunks, 
            model=model_name,
        )