"""
mental_pipeline.py —  ⼼理辅导  RAG  流⽔线 
在标准  RAG  流程基础上增加 ：
1.  情绪识别 （   每次提问前先判断⽤户状态 ）
2.  危机⼲预 （   检测到极端情绪直接⾛安全回复 ）
3.  温暖化⽣成 （   使⽤专属   Prompt ，   temperature   更⾼ ）
4.  多轮记忆 （  同  main   RAG ）
"""
import os 
import   json 
import   time
from typing import List,  Optional 
from   dataclasses   import   dataclass
from   openai   import   OpenAI
from rich.console import  Console 
from rich.panel import Panel 
from rich.table import Table
from mental_config import MENTAL_CoNFIG,  MentalRAGConfig 
from mental_prompt import (
    MENTAL_SYSTEM_PRoMPT,  
    MENTAL_ANSWER_PRoMPT, 
    EMoTIoN_DETECT_PRoMPT, 
    CRISIS_RESPoNSE
)
from document_processor import DocumentProcessor,  Document 
from hybrid_search import HybridRetriever
from   reranker   import   Reranker
from context_compressor import  ContextCompressor 
from query_rewriter import QueryRewriter

console   =   Console()

#   ─────────────────────────────────────────────
#  情绪识别模块
#   ─────────────────────────────────────────────
class   EmotionDetector:  
    """
    ⽤  DeepSeek   快速判断⽤户情绪状态
    识别结果影响后续处理 ：  危机状态  →  直接⼲预 ，  普通状态  →  正常  RAG
    """
    def     __init__   (self,   config:   MentalRAGConfig):
        import httpx 
        self.client   =   OpenAI(
            api_key=config.deepseek_api_key, 
            base_url=config.generator.deepseek_base_url, 
            http_client=httpx.Client(verify=False),
        )
        self.model   =   config.generator.deepseek_model
        
    def   detect(self,   text:   str)   ->   dict:
        """ 返回  {\"emotion\": ..., \"severity\": ..., \"crisis\": bool, \"keywords\": [...]} """
        try:
            prompt =  EMoTIoN_DETECT_PRoMPT.format(text=text) 
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content":  prompt}],  
                max_tokens=150,
                temperature=0.1,   
                response_format={"type":  "json_object"},
            )
            result =  json.loads(response.choices[0].message.content) 
            return result
        except   Exception:
            #  识别失败时返回默认值 ，  不影响主流程
            return   {"emotion":  " 未知 ",  "severity":  " 轻微 ",  "crisis":   False,   "keywords": []}

#   ─────────────────────────────────────────────
#  ⼼理辅导专属⽣成器
#   ─────────────────────────────────────────────
class   MentalGenerator:
    """ 使⽤专属  Prompt   和更⾼  temperature   ⽣成有温度的回复 """
    def    __init__   (self, config:  MentalRAGConfig): 
        import httpx
        self.client = OpenAI(  
            api_key=config.deepseek_api_key, 
            base_url=config.generator.deepseek_base_url, 
            http_client=httpx.Client(verify=False),
        )
        self.cfg   =   config.generator
        MAX_HISToRY  =  8   #  ⼼理对话保留更多历史 （   4 轮 ）
        
    def   generate(self,   question:   str,   context:   str,   history:   List[dict])   ->   str:
        MAX_HISToRY   =  8
        messages = [{"role": "system", "content":  MENTAL_SYSTEM_PRoMPT}]  
        messages.extend(history[-MAX_HISToRY:])
        if   context.strip():
            prompt = MENTAL_ANSWER_PRoMPT.format(context=context,  question=question)
        else:
            #  知识库没有相关内容时 ，  只做情感⽀持
            prompt  =  f" ⽤户说 ：   {question}\\n\\n 请⽤温暖的⽅式回应 ，  以情感⽀持为主 。   "
        messages.append({"role": "user", "content":  prompt}) 
        response = self.client.chat.completions.create(
            model=self.cfg.deepseek_model, 
            messages=messages, 
            max_tokens=self.cfg.max_tokens, 
            temperature=self.cfg.temperature,
        )
        return   response.choices[0].message.content

#   ─────────────────────────────────────────────
#  ⼼理辅导主流⽔线
#   ─────────────────────────────────────────────
class   MentalPipeline:
    def    __init__   (self, config: MentalRAGConfig =  MENTAL_CoNFIG): 
        self.config = config
        self.processor = DocumentProcessor(config) 
        self.retriever = HybridRetriever(config) 
        self.reranker = Reranker(config) 
        self.compressor = ContextCompressor(config) 
        self.query_rewriter = QueryRewriter(config) 
        self.emotion_detector =  EmotionDetector(config) 
        self.generator = MentalGenerator(config) 
        self._index_built = False
        
    def   load_index(self):
        docs_path = os.path.join(self.config.storage_dir,  "docs.json") 
        if not os.path.exists(docs_path):
            console.print(Panel(
                "[bold   red]\\n⼼理知识库不存在 ！   [/bold   red]\\n\\n" 
                " 请先运⾏ ：   \\n"
                "   [bold yellow]python build_mental_kb.py build --dir  ./mental_docs[/bold yellow]",
                border_style="red"
            ))
            raise FileNotFoundError(" ⼼理知识库未建⽴ ") 
        docs = self.processor.load_docs(docs_path)  
        self.retriever.load(docs)
        self._index_built   =   True
        console.print("[green] ✓  ⼼理知识库加载完成 [/green]")
        
    def query(self, question: str, history: List[dict] = None) ->  str:  
        """
        完整流程 ：
        1.  情绪识别
        2.  危机检测  →  直接⼲预
        3.  Query   Rewrite   +   Hybrid   Search   +   Rerank
        4.  有温度的⽣成
        """
        history   =   history   or   []
        # ── Step   1:   情绪识别  ──────────────────
        emotion =  self.emotion_detector.detect(question) 
        emotion_label = emotion.get("emotion", " 未知 ") 
        severity = emotion.get("severity", " 轻微 ")
        is_crisis   =   emotion.get("crisis",   False)
        console.print(
            f"\\n[dim] 情绪识别 ：   {emotion_label} （   {severity} ）   "
            + ("   [bold   red] ⚠  危机信号 [/bold   red]"   if   is_crisis   else  "") 
            + "[/dim]"
        )
        # ── Step   2:   危机⼲预  ──────────────────
        if   is_crisis:
            console.print(Panel( 
                CRISIS_RESPoNSE,
                title="[bold   red] 暖⼼ [/bold   red]",
                border_style="red",
            ))
            return   CRISIS_RESPoNSE
            
        # ──  Step   3:   检索相关知识  ──────────────
        context   =   ""  
        try:
            rewrite_result =  self.query_rewriter.rewrite(question) 
            all_queries = rewrite_result.all_queries()
            candidates =  self.retriever.multi_query_search(all_queries) 
            if candidates:
                ranked_docs =  self.reranker.get_top_docs(rewrite_result.main_query, candidates) 
                compressed = self.compressor.compress(rewrite_result.main_query, ranked_docs) 
                context = self.compressor.format_context(compressed)
        except   Exception   as   e:
            console.print(f"[dim] 检索时遇到问题 ，  将纯⽤情感⽀持回复 :  {e}[/dim]")
            
        # ── Step   4:   ⽣成有温度的回复  ──────────
        answer   =   self.generator.generate(question,   context,   history)
        
        #  打印回复
        console.print(Panel( 
            answer,
            title="[bold   green] 暖⼼ [/bold   green]",
            border_style="green",
        ))
        return   answer

#   ─────────────────────────────────────────
#  对话主循环
#   ─────────────────────────────────────────
    def chat(self):  
        console.print(Panel(
            "[bold   green]\n暖⼼⼼理辅导助⼿已就绪 [/bold   green]\n\n"
            " 你好 ，我是暖⼼ ，很⾼兴认识你\n"
            " 不管你现在是什么⼼情 ，都可以跟我说说 。\n\n"
            " 输⼊[yellow]clear[/yellow] 开始新话题 ， 输⼊[yellow]exit[/yellow] 退出 ",
            border_style="green",
        ))
        history   =   [] 
        while True:
            try:
                user_input  =  input("\n 你 ： ").strip() 
                if not user_input:
                    continue
                if user_input.lower() in {"exit", "quit", "q"}: 
                    console.print("\n[green] 暖⼼ ：   保重 ，   有需要随时来找我\n[/green]")  
                    break
                if user_input.lower() ==  "clear":  
                    history.clear()
                    console.print("[cyan] ✓  已开始新话题 [/cyan]")
                    continue
                answer = self.query(user_input,  history=history) 
                #  存⼊历史
                history.append({"role": "user", "content": user_input}) 
                history.append({"role": "assistant", "content":  answer}) 
                round_num = len(history) // 2
                console.print(f"[dim] （ 已记忆  {round_num}  轮对话 ）  [/dim]")
            except   KeyboardInterrupt:
                console.print("\\n[green] 暖⼼ ： 保重 ， 有需要随时来找我\n[/green]")  
                break
            except Exception as e: 
                console.print(f"[red] 出错了 :  {e}[/red]")