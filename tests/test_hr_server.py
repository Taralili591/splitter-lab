"""Local HTTP contract tests; no real credentials or cloud API calls."""
import http.client
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from unittest.mock import patch

from labs.hr_chunking_lab.engine import LabEngine, LAB_DIR
from labs.hr_chunking_lab.server import Application, make_server


class HRServerTests(unittest.TestCase):
    def setUp(self):
        self.environment=patch.dict(os.environ,{},clear=True)
        self.environment.start()
        self.directory=TemporaryDirectory()
        self.root=Path(self.directory.name)
        self.lab=self.root/'lab'
        shutil.copytree(LAB_DIR/'data',self.lab/'data')
        (self.lab/'web').mkdir()
        (self.lab/'web/index.html').write_text('<html><title>HR lab</title></html>', encoding='utf-8')
        self.app=Application(self.lab,self.root,LabEngine(self.lab/'data',self.lab/'runtime'))
        self.server=make_server(0,self.app)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.port=self.server.server_port

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join(2)
        self.directory.cleanup();self.environment.stop()

    def request(self,method,path,payload=None,origin=None):
        conn=http.client.HTTPConnection('127.0.0.1',self.port,timeout=5)
        headers={}
        if origin:headers['Origin']=origin
        data=None
        if payload is not None:
            headers['Content-Type']='application/json'
            data=json.dumps(payload).encode()
        conn.request(method,path,data,headers)
        response=conn.getresponse()
        raw=response.read();status=response.status;ctype=response.getheader('Content-Type')
        conn.close()
        return status,json.loads(raw) if ctype and ctype.startswith('application/json') else raw

    def completed(self,jid):
        for _ in range(150):
            status,job=self.request('GET','/api/jobs/'+jid)
            self.assertEqual(status,200)
            if job['status']!='running':return job
            time.sleep(.01)
        self.fail('local job did not finish')

    def test_bootstrap_static_and_config_do_not_return_credentials(self):
        self.assertEqual(self.request('GET','/api/health'),(200,{'app':'hr-chunking-lab','version':1}))
        status,boot=self.request('GET','/api/bootstrap')
        self.assertEqual(status,200)
        self.assertEqual(len(boot['questions']),12)
        self.assertFalse(boot['config']['deepseek_key_set'])
        self.assertEqual(self.request('GET','/')[0],200)
        secret='sk-test-http-private'
        status,data=self.request('POST','/api/config',{'deepseek_key':secret})
        self.assertEqual(status,200)
        self.assertNotIn(secret,json.dumps(data))
        self.assertTrue(data['config']['deepseek_key_set'])
        self.assertFalse((self.root/'.env').exists())

    def test_preview_is_local_and_semantic_does_not_fake_preview(self):
        with patch('requests.post') as remote:
            status,result=self.request('POST','/api/preview',{'strategy':'fixed'})
            self.assertEqual(status,200);self.assertTrue(result['chunks'])
            status,result=self.request('POST','/api/preview',{'strategy':'semantic'})
            self.assertTrue(result['needs_build']);self.assertEqual(result['chunks'],[])
            remote.assert_not_called()

    def test_boundary_demo_is_source_only_and_needs_no_config_or_models(self):
        with patch('requests.post') as remote, patch.object(self.app.config,'get') as config:
            status,result=self.request('POST','/api/boundary-demo',{'sample':'short','chunk_size':280})
            self.assertEqual(status,200)
            self.assertEqual([method['chunk_count'] for method in result['methods']],[1,2])
            self.assertEqual(result['params']['chunk_overlap'],0)
            self.assertFalse(result['controls']['title_prefix_added'])
            self.assertEqual(self.app.jobs,{})
            remote.assert_not_called();config.assert_not_called()

    def test_boundary_demo_rejects_other_experiment_settings(self):
        for payload in ({'sample':'missing'},{'chunk_size':True},{'chunk_size':601},
                        {'chunk_overlap':50},{'params':{'top_k':3}}):
            with self.subTest(payload=payload):
                status,result=self.request('POST','/api/boundary-demo',payload)
                self.assertEqual(status,400)
                self.assertIn('error',result)

    def test_keyword_job_export_and_budget(self):
        q=self.app.engine.questions()[0]
        with patch('requests.post') as remote:
            status,started=self.request('POST','/api/retrieve',{
                'strategies':['fixed','parent_child'],'question':q['question'],'question_id':q['id'],
                'params':{'retrieval_mode':'keyword','context_budget_chars':800}})
            self.assertEqual(status,202)
            job=self.completed(started['job_id'])
            self.assertEqual(job['status'],'complete',job)
            result=job['result']
            self.assertEqual(len(result['results']),2)
            self.assertTrue(all(x['context_chars']<=800 for x in result['results']))
            status,export=self.request('GET','/api/export/'+result['run_id'])
            self.assertEqual(status,200);self.assertEqual(export['question_id'],'Q01')
            remote.assert_not_called()

    def test_bad_origin_and_invalid_requests_do_not_change_config(self):
        status,_=self.request('POST','/api/config',{'deepseek_key':'fake'},origin='https://unrelated.example')
        self.assertEqual(status,403);self.assertFalse((self.lab/'.env').exists())
        self.assertEqual(self.request('POST','/api/preview',[])[0],400)
        self.assertEqual(self.request('GET','/api/export/.env')[0],400)

    def test_missing_key_job_returns_clear_error_without_network(self):
        with patch('requests.post') as remote:
            status,started=self.request('POST','/api/retrieve',{'strategies':['fixed'],'question':'请假要怎么申请？'})
            self.assertEqual(status,202)
            job=self.completed(started['job_id'])
            self.assertEqual(job['status'],'error')
            self.assertIn('API',job['error'])
            remote.assert_not_called()

    def test_method_specific_params_reach_retrieval_and_export(self):
        overrides={'fixed':{'chunk_size':160,'chunk_overlap':20},
                   'recursive':{'chunk_size':280,'chunk_overlap':0}}
        with patch('requests.post') as remote:
            status,started=self.request('POST','/api/retrieve',{
                'strategies':['fixed','recursive'],'question':'公司福利年假怎么申请？',
                'params':{'retrieval_mode':'keyword','top_k':1,'context_budget_chars':600},
                'strategy_params':overrides})
            self.assertEqual(status,202)
            job=self.completed(started['job_id'])
            self.assertEqual(job['status'],'complete',job)
            run=job['result']
            self.assertEqual(run['strategy_params'],overrides)
            self.assertEqual([item['params']['chunk_size'] for item in run['results']],[160,280])
            self.assertTrue(all(item['included_count']<=1 and item['context_chars']<=600 for item in run['results']))
            status,export=self.request('GET','/api/export/'+run['run_id'])
            self.assertEqual(status,200)
            self.assertEqual(export['strategy_params'],overrides)
            remote.assert_not_called()

    def test_invalid_method_specific_params_are_rejected_before_job(self):
        with patch('requests.post') as remote:
            status,data=self.request('POST','/api/build',{
                'strategies':['fixed'],'strategy_params':{'fixed':{'top_k':2}}})
            self.assertEqual(status,400)
            self.assertIn('专用参数',data['error'])
            self.assertEqual(self.app.jobs,{})
            remote.assert_not_called()

    def test_contextual_cache_preflight_uses_own_method_settings(self):
        from tests.test_hr_engine import RecordingChat
        overrides={'contextual':{'chunk_size':160,'chunk_overlap':20}}
        params={'retrieval_mode':'keyword'}
        with patch('labs.hr_chunking_lab.engine.DeepSeekChat',RecordingChat), patch('requests.post') as remote:
            self.app.engine.build(['contextual'],params,self.app.config.get(),strategy_params=overrides)
            status,started=self.request('POST','/api/build',{
                'strategies':['contextual'],'params':params,'strategy_params':overrides})
            self.assertEqual(status,202,started)
            self.assertEqual(self.completed(started['job_id'])['status'],'complete')
            remote.assert_not_called()


if __name__=='__main__':unittest.main()
