"""
query_rewriter.py  —  查询改写层
⽀持 ：   本地  Qwen3-7B  /  Anthropic   API  /  openAI   API
输出 ：   main_query  +  sub_queries （   JSoN   格式 ）
"""
import   json 
import re
from dataclasses import  dataclass 
from typing import List
from   rich.console   import   Console
from rag_config import RAGConfig,  DEFAULT_CONFIG 

console = Console()

REWRITE_PRoMPT   =   """ 你是⼀个专业的问题改写助⼿ ，   请将⽤户问题改写为更适合检索的形式 。
要求 ：
1.  main_query ：   将问题改写得更清晰 、   结构化 ，   补充可能缺失的上下⽂
2.  sub_queries ：   ⽣成   {num}   个语义相关但表达⽅式不同的查询 （   换词 、   换⻆度 、   换问法 ）
⽤户问题 ：   {question}
请严格输出   JSoN ，   不要输出任何其他内容 ：
{{
"main_query":  " 改写后的主查询 ",
"sub_queries":  [\" ⼦查询 1\",  \" ⼦查询 2\",  \" ⼦查询 3\"]
}}"""

@dataclass
class RewriteResult: 
    main_query: str 
    sub_queries:   List[str]
    
    def   all_queries(self)   ->   List[str]:
        return   [self.main_query]   +   self.sub_queries

#   ─────────────────────────────────────────────
#  解析  LLM  输出为结构化结果
#   ─────────────────────────────────────────────
def _parse_rewrite_output(text: str, original: str, num: int) ->  RewriteResult: 
    """ 健壮解析  LLM JSoN  输出 ，  失败时返回原始查询 """
    #  尝试提取  JSoN   块
    match = re.search(r"\{.*?\}", text,  re.DOTALL) 
    if match:
        try:
            data   =   json.loads(match.group())
            main_q = data.get("main_query",  original).strip() 
            sub_qs = data.get("sub_queries", [])[:num]
            #  补⻬不⾜的  sub_queries
            while len(sub_qs) < num:  
                sub_qs.append(original)
            return RewriteResult(main_query=main_q,  sub_queries=sub_qs) 
        except json.JSONDecodeError:
            pass
    #  解析失败 ，  返回原始查询
    console.print("[yellow] ⚠  Query   Rewrite   解析失败 ，  使⽤原始查询 [/yellow]")
    return   RewriteResult(main_query=original,   sub_queries=[original]   *   num)

#   ─────────────────────────────────────────────
#  DeepSeek   API   改写 （  推荐 ，  中⽂理解最强 ）
#   ─────────────────────────────────────────────
class   DeepSeekQueryRewriter:  
    """
    DeepSeek   API   完全兼容  openAI   SDK
    只需设置  base_url   和  api_key ， 其余与  openAI   调⽤⽅式完全相同 
    价格约为  GPT-4o  的  1/30 ， 中⽂  Query Rewrite  效果优秀
    """
    def    __init__   (self, config: RAGConfig =  DEFAULT_CONFIG): 
        import httpx
        from   openai   import   OpenAI 
        self.client = OpenAI(
            api_key=config.deepseek_api_key, 
            base_url=config.query_rewriter.deepseek_base_url, 
            http_client=httpx.Client(verify=False),
        )
        self.cfg   =   config.query_rewriter
        
    def   rewrite(self,   question:   str)   ->   RewriteResult:
        prompt = REWRITE_PRoMPT.format(question=question,  num=self.cfg.num_sub_queries)
        response = self.client.chat.completions.create(
            model=self.cfg.deepseek_model,  
            messages=[{"role": "user", "content":  prompt}],  
            max_tokens=512,
            temperature=0.3,
            response_format={"type":   "json_object"},   #  DeepSeek   ⽀持  JSoN   模式
        )
        text   =   response.choices[0].message.content
        return _parse_rewrite_output(text, question,  self.cfg.num_sub_queries) 

class LocalQueryRewriter:
    def    __init__   (self, config: RAGConfig =  DEFAULT_CONFIG): 
        self.cfg = config.query_rewriter
        self.tokenizer   =   None 
        self.model = None
        
    def   _load(self):
        if   self.model   is   not   None:  
            return
        from transformers import AutoTokenizer,  AutoModelForCausalLM 
        import torch
        console.print(f"[cyan] 加载  Query   Rewriter   模型 :  {self.cfg.local_model}[/cyan]")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.local_model,  trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.local_model, 
            device_map=self.cfg.local_device, 
            torch_dtype=torch.float16, 
            trust_remote_code=True,
        )
        
    def rewrite(self, question: str) ->  RewriteResult:  
        self._load()
        import   torch
        prompt = REWRITE_PRoMPT.format(question=question,  num=self.cfg.num_sub_queries)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.cfg.local_device)
        with torch.no_grad():
            outputs   =   self.model.generate(
                **inputs, 
                max_new_tokens=512, 
                temperature=0.3, 
                do_sample=True,
            )
        text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],  skip_special_tokens=True)
        return _parse_rewrite_output(text, question, self.cfg.num_sub_queries)

#   ─────────────────────────────────────────────
#  Anthropic   API   改写
#   ─────────────────────────────────────────────
class   AnthropicQueryRewriter:
    def    __init__   (self, config: RAGConfig =  DEFAULT_CONFIG): 
        self.cfg = config.query_rewriter
        self.api_key   =   config.anthropic_api_key
        
    def rewrite(self, question: str) ->  RewriteResult: 
        import anthropic
        client   =   anthropic.Anthropic(api_key=self.api_key)
        prompt = REWRITE_PRoMPT.format(question=question,  num=self.cfg.num_sub_queries)
        #  使⽤  Haiku ：  改写任务简单 ，   Haiku   速度快 、  成本低
        response = client.messages.create(  
            model=self.cfg.anthropic_model,      #   claude-haiku-4-5 
            max_tokens=512,
            messages=[{"role":   "user",   "content":   prompt}],
        )
        text   =   response.content[0].text
        return   _parse_rewrite_output(text,   question,   self.cfg.num_sub_queries)

#   ─────────────────────────────────────────────
#  openAI   API   改写
#   ─────────────────────────────────────────────
class   OpenAIQueryRewriter:
    def    __init__   (self, config: RAGConfig =  DEFAULT_CONFIG): 
        self.cfg = config.query_rewriter
        self.api_key   =   config.openai_api_key
        
    def rewrite(self, question: str) ->  RewriteResult: 
        from openai import OpenAI
        client   =   OpenAI(api_key=self.api_key)
        prompt = REWRITE_PRoMPT.format(question=question,  num=self.cfg.num_sub_queries)
        response = client.chat.completions.create(
            model=self.cfg.openai_model,
            messages=[{"role": "user", "content":  prompt}],  
            max_tokens=512,
            temperature=0.3,   
            response_format={"type":  "json_object"},
        )
        text   =   response.choices[0].message.content
        return   _parse_rewrite_output(text,   question,   self.cfg.num_sub_queries)

#   ─────────────────────────────────────────────
#  ⼯⼚函数 （  统⼀⼊⼝ ）
#   ─────────────────────────────────────────────
class   QueryRewriter:  
    """
    统⼀接⼝ ，   根据  config.query_rewriter.api_provider   ⾃动路由 ：
    "deepseek"  →  DeepSeek   API （  默认 ，  中⽂强 、  成本低 ）
    "anthropic"   →   Claude   Haiku
    "openai"     → GPT-4o-mini  
    "local"   →  本地  Qwen3-7B
    """
    def    __init__   (self, config: RAGConfig =  DEFAULT_CONFIG): 
        self.cfg = config.query_rewriter
        provider = self.cfg.api_provider if self.cfg.mode == "api" else  "local" 
        if provider == "deepseek":
            self._backend =  DeepSeekQueryRewriter(config) 
        elif provider == "anthropic":
            self._backend =  AnthropicQueryRewriter(config) 
        elif provider == "openai":
            self._backend =  OpenAIQueryRewriter(config)  
        else:
            self._backend   =   LocalQueryRewriter(config)
            
    def   rewrite(self,   question:   str)   ->   RewriteResult:
        console.print(f"\n[bold cyan]\nQuery Rewrite[/bold cyan]: {question}")
        result = self._backend.rewrite(question)
        console.print(f"   main_query:  [green]{result.main_query}[/green]")
        for i, q in enumerate(result.sub_queries):
            console.print(f"   sub_query [{i+1}]:  {q}")
        return result