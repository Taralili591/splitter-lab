"""Local-only browser lab; long-running model operations use observable jobs."""
from __future__ import annotations

import argparse
import copy
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
import uuid
import webbrowser

from .engine import DEFAULT_PARAMS, LAB_DIR, STRATEGIES, LabEngine, resolve_strategy_params, validate_strategies
from .providers import ConfigStore, DeepSeekChat, ProviderError, SiliconFlowEmbedder
from .boundary_demo import boundary_demo

ROOT_DIR=LAB_DIR.parents[1]


class Application:
    def __init__(self, lab_dir=LAB_DIR, root_dir=ROOT_DIR, engine=None):
        self.lab_dir=Path(lab_dir)
        self.config=ConfigStore(self.lab_dir,root_dir)
        self.engine=engine or LabEngine(self.lab_dir/'data',self.lab_dir/'runtime')
        self.jobs={}
        self.lock=threading.Lock()

    def public_config(self):
        config=self.config.public()
        return {**config, 'deepseek_key_set':config['deepseek_configured'],
                'embedding_key_set':config['siliconflow_configured'],
                'embedding_model':config['siliconflow_embedding_model'],
                'embedding_base_url':config['siliconflow_base_url'],
                'config_path':str(self.lab_dir/'.env')}

    def bootstrap(self):
        doc=self.engine.document()
        return {'config':self.public_config(),'strategies':STRATEGIES,'default_params':DEFAULT_PARAMS,
                'questions':[{'id':q['id'],'label':q['id']+' '+q['question'][:40],
                              'question':q['question']} for q in self.engine.questions()],
                'document_count':1,'document':{'title':doc.title,'text':doc.text,
                                             'clause_count':len(doc.clauses)},
                'query_date':'2026-11-12'}

    def save_config(self,payload):
        aliases={'deepseek_key':'deepseek_api_key','embedding_key':'siliconflow_api_key',
                 'embedding_model':'siliconflow_embedding_model','embedding_base_url':'siliconflow_base_url'}
        allowed={'deepseek_key','embedding_key','deepseek_model','embedding_model',
                 'deepseek_base_url','embedding_base_url'}
        updates={aliases.get(k,k):v for k,v in payload.items() if k in allowed}
        self.config.save(updates)
        return {'config':self.public_config()}

    def test_connection(self,payload):
        settings=self.config.get()
        provider=payload.get('provider')
        if provider=='deepseek':
            result=DeepSeekChat(settings).chat([{'role':'user','content':'仅回复 OK。'}],max_tokens=20)
            return {'ok':True,'message':f'DeepSeek 连接成功，模型 {result["model"]}。'}
        if provider=='embedding':
            # A unique short input tests the configured API rather than merely
            # proving that a previously generated embedding exists on disk.
            client=SiliconFlowEmbedder(settings,self.lab_dir/'runtime'/'embedding_cache')
            vec=client.embed(['连接测试 '+uuid.uuid4().hex[:8]],role='query')[0]
            return {'ok':True,'message':f'硅基流动连接成功，返回 {len(vec)} 维向量。'}
        raise ValueError('请选择 deepseek 或 embedding。')

    def start_job(self,action,payload):
        settings=self.config.get()
        if action in {'build','retrieve'}:
            strategies=validate_strategies(payload.get('strategies',['recursive']))
            params,resolved=resolve_strategy_params(strategies,payload.get('params'),payload.get('strategy_params'))
            if 'contextual' in strategies:
                doc=self.engine.document()
                key=self.engine._key(doc,'contextual',resolved['contextual'],settings)
                cached=(self.engine.runtime_dir/'indexes'/(key+'.json')).is_file()
                if not cached and not settings.get('deepseek_api_key'):
                    raise ValueError('上下文补充首次构建需要 DeepSeek API Key，请先填写「连接与密钥」。')
                if not cached and params['retrieval_mode']=='vector' and not settings.get('siliconflow_api_key'):
                    raise ValueError('向量索引还需要硅基流动 API Key；请先填写，避免生成背景后才中断。')
        elif action=='answer':
            strategies=validate_strategies(payload.get('strategies',['recursive']))
        else:
            raise ValueError('未知实验操作。')
        with self.lock:
            if any(job['status']=='running' for job in self.jobs.values()):
                raise ValueError('已有实验正在运行，请等待本轮完成。')
            jid=uuid.uuid4().hex[:12]
            self.jobs[jid]={'job_id':jid,'status':'running','progress':['开始本轮实验。'],'result':None}
            # Completed jobs are also persisted as run records; keep only a
            # bounded polling history in server memory.
            if len(self.jobs)>80:
                for old in list(self.jobs):
                    if old!=jid and self.jobs[old]['status']!='running':
                        self.jobs.pop(old)
                        break

        def progress(message):
            with self.lock:
                self.jobs[jid]['progress'].append(message)

        def work():
            try:
                if action=='build':
                    result=self.engine.build(strategies,payload.get('params'),settings,progress,
                                             strategy_params=payload.get('strategy_params'))
                elif action=='retrieve':
                    result=self.engine.retrieve(strategies,payload.get('question',''),payload.get('params'),settings,
                                                payload.get('question_id'),progress,
                                                strategy_params=payload.get('strategy_params'))
                else:
                    result=self.engine.answer(payload.get('run_id'),strategies,settings,progress)
                with self.lock:
                    self.jobs[jid].update(status='complete',result=result)
                    self.jobs[jid]['progress'].append('本轮完成。')
            except Exception as error:
                message=str(error)
                for name in ('deepseek_api_key','siliconflow_api_key'):
                    if settings.get(name): message=message.replace(settings[name],'<redacted>')
                with self.lock:
                    self.jobs[jid].update(status='error',error=message[:700] or '实验未完成。')
                    self.jobs[jid]['progress'].append('实验停止；没有用模拟向量或预设回答代替失败的 API。')

        threading.Thread(target=work,daemon=True).start()
        return {'job_id':jid}

    def job(self,jid):
        with self.lock:
            return copy.deepcopy(self.jobs.get(jid))


def handler_for(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_):
            pass

        def reply(self,status,value,content_type='application/json; charset=utf-8',download=None):
            raw=value if isinstance(value,bytes) else json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(raw)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Referrer-Policy','no-referrer')
            if download: self.send_header('Content-Disposition',f'attachment; filename="{download}"')
            self.end_headers()
            try: self.wfile.write(raw)
            except (BrokenPipeError,ConnectionResetError): pass

        def local(self):
            try: host=urlsplit('http://'+self.headers.get('Host','')).hostname
            except ValueError: return False
            origin=self.headers.get('Origin')
            origins={f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'}
            return host in {'127.0.0.1','localhost'} and (not origin or origin in origins)

        def do_GET(self):
            if not self.local():
                self.reply(403,{'error':'仅允许本机实验页面访问。'});return
            path=urlsplit(self.path).path
            try:
                if path=='/':
                    self.reply(200,(app.lab_dir/'web/index.html').read_bytes(),'text/html; charset=utf-8')
                elif path=='/api/health': self.reply(200,{'app':'hr-chunking-lab','version':1})
                elif path=='/api/bootstrap': self.reply(200,app.bootstrap())
                elif path.startswith('/api/jobs/'):
                    job=app.job(path.rsplit('/',1)[-1])
                    self.reply(200 if job else 404,job or {'error':'未找到本轮任务，请重新运行。'})
                elif path.startswith('/api/export/'):
                    rid=path.rsplit('/',1)[-1]
                    self.reply(200,app.engine.load_run(rid),download=f'hr-rag-{rid}.json')
                elif path=='/favicon.ico': self.reply(204,b'','image/x-icon')
                else:self.reply(404,{'error':'页面不存在。'})
            except (ValueError,OSError,ProviderError) as error:
                self.reply(400,{'error':str(error)[:600]})

        def do_POST(self):
            if not self.local() or not self.headers.get('Content-Type','').startswith('application/json'):
                self.reply(403,{'error':'请从本机页面发送 JSON 请求。'});return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=24000: raise ValueError('请求体为空或过大。')
                payload=json.loads(self.rfile.read(length))
                if not isinstance(payload,dict): raise ValueError('请求参数应为对象。')
                path=urlsplit(self.path).path
                if path=='/api/config':self.reply(200,app.save_config(payload))
                elif path=='/api/test':self.reply(200,app.test_connection(payload))
                elif path=='/api/preview':self.reply(200,app.engine.preview(payload.get('strategy','recursive'),payload.get('params'),app.config.get()))
                elif path=='/api/boundary-demo':
                    if set(payload)-{'sample','chunk_size'}:
                        raise ValueError('切口对照只接受样例和块长；重叠固定为 0，不使用检索参数。')
                    self.reply(200,boundary_demo(payload.get('sample','short'),payload.get('chunk_size',280),
                                                 data_dir=app.lab_dir/'data'))
                elif path in {'/api/build','/api/retrieve','/api/answer'}:
                    self.reply(202,app.start_job(path.rsplit('/',1)[-1],payload))
                else:self.reply(404,{'error':'接口不存在。'})
            except (ValueError,TypeError,OSError,ProviderError) as error:
                self.reply(400,{'error':str(error)[:600]})

    return Handler


def make_server(port=8768,app=None):
    return ThreadingHTTPServer(('127.0.0.1',port),handler_for(app or Application()))


def main():
    parser=argparse.ArgumentParser(description='员工手册 RAG 六种切块实验室')
    parser.add_argument('--port',type=int,default=8768)
    parser.add_argument('--open',action='store_true',help='启动后打开本机浏览器')
    args=parser.parse_args()
    try: server=make_server(args.port)
    except OSError:
        if args.open:
            connection=None
            try:
                connection=http.client.HTTPConnection('127.0.0.1',args.port,timeout=2)
                connection.request('GET','/api/health')
                response=connection.getresponse()
                health=json.loads(response.read())
                if response.status==200 and isinstance(health,dict) and health.get('app')=='hr-chunking-lab':
                    url=f'http://127.0.0.1:{args.port}'
                    print(f'员工手册实验室已在运行：{url}',flush=True)
                    webbrowser.open(url)
                    return
            except (OSError,ValueError,http.client.HTTPException):
                pass
            finally:
                if connection is not None:connection.close()
        parser.exit(1,f'端口 {args.port} 无法使用，请用 --port 8769 换一个端口。\n')
    url=f'http://127.0.0.1:{server.server_port}'
    print(f'员工手册 RAG 实验室：{url}\nAPI 配置：{LAB_DIR / ".env"}\n按 Ctrl+C 停止。',flush=True)
    if args.open:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
