"""HR实验适配器验证：仅临时配置和mock HTTP，不读取用户密钥或调用远端。"""
import json
from contextlib import closing
import os
import sqlite3
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import requests

from labs.hr_chunking_lab.providers import (
    ConfigStore, DEFAULTS, DeepSeekChat, ENV_NAMES, ProviderError,
    SiliconFlowEmbedder, TIMEOUT,
)


DS_KEY = "test-only-ds-key-not-a-real-credential"
SF_KEY = "test-only-sf-key-not-a-real-credential"


def settings(**updates):
    return {**DEFAULTS, "deepseek_api_key": DS_KEY, "siliconflow_api_key": SF_KEY, **updates}


def response(payload=None, status=200, text=""):
    result = Mock(status_code=status, text=text)
    result.json.return_value = payload
    return result


def embeddings(values):
    return response({"data": [{"index": i, "embedding": v} for i, v in enumerate(values)]})


class HrConfigTests(unittest.TestCase):
    def test_defaults_do_not_require_existing_keys(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            store = ConfigStore(Path(directory) / "lab", directory)
            self.assertEqual(store.get(), DEFAULTS)
            self.assertEqual(store.public()["configured"], {"deepseek": False, "siliconflow": False})
            self.assertFalse(store.path.exists())

    def test_root_sf_compatibility_local_and_process_precedence(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            lab = root / "lab"
            lab.mkdir()
            (root / ".env").write_text(
                'SILICONFLOW_API_KEY="root-sf"\nRAG_LAB_BASE_URL="https://root.example/v1"\n'
                'RAG_LAB_EMBEDDING_MODEL="root-embed"\nDEEPSEEK_API_KEY="do-not-import"\n'
                'UNRELATED_SECRET="never-export"\n', encoding="utf-8")
            (lab / ".env").write_text('SILICONFLOW_API_KEY=""\nSILICONFLOW_EMBEDDING_MODEL="local-embed"\n', encoding="utf-8")
            store = ConfigStore(lab, root)
            first = store.get()
            self.assertEqual(first["siliconflow_api_key"], "root-sf")
            self.assertEqual(first["siliconflow_base_url"], "https://root.example/v1")
            self.assertEqual(first["siliconflow_embedding_model"], "local-embed")
            self.assertEqual(first["deepseek_api_key"], "")
            os.environ.update({"SILICONFLOW_API_KEY": "process-sf", "RAG_LAB_BASE_URL": "https://process.example/v1", "DEEPSEEK_API_KEY": "process-ds"})
            current = store.get()
            self.assertEqual(current["siliconflow_api_key"], "process-sf")
            self.assertEqual(current["siliconflow_base_url"], "https://process.example/v1")
            self.assertEqual(current["deepseek_api_key"], "process-ds")
            public = json.dumps(store.public())
            for secret in ("root-sf", "process-sf", "process-ds", "do-not-import", "never-export"):
                self.assertNotIn(secret, public)

    def test_save_atomic_private_local_file_preserves_blank_keys(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            (root / ".env").write_text('SILICONFLOW_API_KEY="root-only"\n', encoding="utf-8")
            original_root = (root / ".env").read_bytes()
            lab = root / "lab"
            lab.mkdir()
            path = lab / ".env"
            path.write_text('# keep comment\nOTHER_SETTING="keep"\nDEEPSEEK_API_KEY="old-ds"\n', encoding="utf-8")
            store = ConfigStore(lab, root)
            result = store.save({"deepseek_api_key": "", "siliconflow_api_key": "", "deepseek_model": "custom-flash"})
            self.assertEqual(store.get()["deepseek_api_key"], "old-ds")
            self.assertEqual(store.get()["siliconflow_api_key"], "root-only")
            self.assertEqual(store.get()["deepseek_model"], "custom-flash")
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual((root / ".env").read_bytes(), original_root)
            self.assertIn('OTHER_SETTING="keep"', path.read_text())
            self.assertIn('# keep comment', path.read_text())
            for name in ENV_NAMES.values():
                self.assertIn(name + "=", path.read_text())
            self.assertNotIn("api_key", result)
            for key in ("deepseek_api_key", "siliconflow_api_key"):
                self.assertNotIn(key, result)
            self.assertNotIn("old-ds", json.dumps(result))
            self.assertNotIn("root-only", json.dumps(result))
            self.assertEqual(list(lab.glob('.env-write-*')), [])

    def test_save_without_descriptor_chmod_keeps_atomic_updates(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            lab = Path(directory) / "lab"
            store = ConfigStore(lab, directory)
            with patch.object(os, "fchmod", None, create=True), patch.object(os, "chmod", wraps=os.chmod) as chmod:
                public = store.save({"deepseek_api_key": "local-test-key", "deepseek_model": "custom-flash"})
            chmod.assert_called_once()
            self.assertEqual(store.get()["deepseek_api_key"], "local-test-key")
            self.assertEqual(store.get()["deepseek_model"], "custom-flash")
            self.assertNotIn("local-test-key", json.dumps(public))
            self.assertEqual(list(lab.glob('.env-write-*')), [])

    def test_save_creates_all_placeholders_and_process_remains_authoritative(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            store = ConfigStore(Path(directory)/"lab", directory)
            store.save({})
            self.assertEqual(len(store.path.read_text().splitlines()), len(ENV_NAMES))
            os.environ["DEEPSEEK_MODEL"] = "process-model"
            store.save({"deepseek_model": "file-model"})
            self.assertEqual(store.get()["deepseek_model"], "process-model")
            self.assertIn('DEEPSEEK_MODEL="file-model"', store.path.read_text())

    def test_failed_atomic_save_leaves_old_config(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            store = ConfigStore(Path(directory)/"lab", directory)
            store.save({"deepseek_model": "before"})
            old = store.path.read_bytes()
            with patch('labs.hr_chunking_lab.providers.os.replace', side_effect=OSError("private detail")):
                with self.assertRaises(ProviderError) as caught:
                    store.save({"deepseek_model": "after"})
            self.assertEqual(caught.exception.code, "config_write")
            self.assertNotIn("private detail", str(caught.exception))
            self.assertEqual(store.path.read_bytes(), old)
            self.assertEqual(list(store.lab_dir.glob('.env-write-*')), [])

    def test_https_and_header_validation_never_echoes_input(self):
        invalid = ["http://example.com", "https://user:password@example.com", "https://example.com?q=private", "https://example.com/#private", "https://example.com:bad", "https://bad host.example"]
        for value in invalid:
            with self.subTest(url=value), self.assertRaises(ProviderError) as caught:
                DeepSeekChat(settings(deepseek_base_url=value))
            self.assertNotIn(value, str(caught.exception))
        with self.assertRaises(ProviderError):
            DeepSeekChat(settings(deepseek_api_key=DS_KEY + '\nInjected: yes'))


class HrEmbeddingTests(unittest.TestCase):
    def test_reorders_batches_deduplicates_and_caches_without_plaintext(self):
        with TemporaryDirectory() as directory:
            client = SiliconFlowEmbedder(settings(), directory)
            texts = [f'private-document-{i}' for i in range(18)] + ['private-document-0']

            def remote(url, **kwargs):
                self.assertEqual(url, 'https://api.siliconflow.cn/v1/embeddings')
                self.assertEqual(kwargs['headers']['Authorization'], 'Bearer ' + SF_KEY)
                self.assertEqual(kwargs['timeout'], TIMEOUT)
                self.assertFalse(kwargs['allow_redirects'])
                batch = kwargs['json']['input']
                self.assertLessEqual(len(batch), 16)
                rows = [{"index":i,"embedding":[float(text.rsplit('-',1)[1]),1]} for i,text in enumerate(batch)]
                return response({"data":list(reversed(rows)),"usage":{"prompt_tokens":len(batch)}})

            with patch('requests.post', side_effect=remote) as post:
                result = client.embed(texts)
                self.assertEqual(result, [[float(i),1.0] for i in range(18)]+[[0.0,1.0]])
                self.assertEqual(post.call_count,2)
                self.assertEqual(client.last_usage, {'prompt_tokens':18})
                self.assertEqual(client.last_stats['dimension'],2)
                second = SiliconFlowEmbedder(settings(), directory)
                self.assertEqual(second.embed(texts),result)
                self.assertEqual(second.last_stats['cache_hits'],19)
                self.assertEqual(second.last_stats['requests'],0)
                self.assertEqual(post.call_count,2)
            data = client.cache_path.read_bytes()
            for private in (DS_KEY,SF_KEY,'private-document-0'):
                self.assertNotIn(private.encode(), data)

    def test_cache_identity_includes_url_model_role_and_exact_text(self):
        with TemporaryDirectory() as directory, patch('requests.post', return_value=embeddings([[1,2]])) as post:
            first = SiliconFlowEmbedder(settings(),directory)
            first.embed(['same'])
            first.embed(['same'])
            self.assertEqual(post.call_count,1)
            first.embed(['same'],role='query')
            SiliconFlowEmbedder(settings(siliconflow_embedding_model='new-model'),directory).embed(['same'])
            SiliconFlowEmbedder(settings(siliconflow_base_url='https://other.example/v1'),directory).embed(['same'])
            first.embed(['same '])
            self.assertEqual(post.call_count,5)

    def test_count_index_dimension_and_finiteness_validation(self):
        cases = [
            (['a'], {'data':[]}),
            (['a'], {'data':[{'index':1,'embedding':[1]}]}),
            (['a'], {'data':[{'index':True,'embedding':[1]}]}),
            (['a'], {'data':[{'embedding':[1]}]}),
            (['a','b'], {'data':[{'index':0,'embedding':[1]},{'index':0,'embedding':[2]}]}),
            (['a','b'], {'data':[{'index':0,'embedding':[1]},{'index':1,'embedding':[2,3]}]}),
            (['a'], {'data':[{'index':0,'embedding':[]}]}),
            (['a'], {'data':[{'index':0,'embedding':[float('nan')]}]}),
            (['a'], {'data':[{'index':0,'embedding':[float('inf')]}]}),
            (['a'], {'data':[{'index':0,'embedding':[True]}]}),
            (['a'], {'data':[{'index':0,'embedding':['1']}]}),
        ]
        for texts,payload in cases:
            with self.subTest(payload=payload),TemporaryDirectory() as directory:
                client=SiliconFlowEmbedder(settings(),directory)
                with patch('requests.post',return_value=response(payload)),self.assertRaises(ProviderError):
                    client.embed(texts)
                with closing(sqlite3.connect(client.cache_path)) as db:
                    self.assertEqual(db.execute('SELECT COUNT(*) FROM embeddings').fetchone()[0],0)

    def test_dimension_consistency_across_batches_and_cached_vectors(self):
        with TemporaryDirectory() as directory:
            client=SiliconFlowEmbedder(settings(),directory)
            with patch('requests.post',side_effect=[embeddings([[1,2]]*16),embeddings([[1,2,3]])]),self.assertRaises(ProviderError):
                client.embed([f'text-{i}' for i in range(17)])
            with patch('requests.post',return_value=embeddings([[1,2,3]])),self.assertRaises(ProviderError):
                client.embed(['text-0','new-text'])

    def test_dimension_consistency_across_document_and_query_calls(self):
        with TemporaryDirectory() as directory:
            client=SiliconFlowEmbedder(settings(),directory)
            with patch('requests.post',return_value=embeddings([[1,2]])):
                client.embed(['document'])
            with patch('requests.post',return_value=embeddings([[1,2,3]])),self.assertRaises(ProviderError):
                client.embed(['question'],role='query')

    def test_invalid_cache_entry_is_recomputed(self):
        with TemporaryDirectory() as directory:
            client=SiliconFlowEmbedder(settings(),directory)
            key=client._cache_key('document','document')
            connection=client._connection()
            with connection:
                connection.execute('INSERT INTO embeddings VALUES (?,?)',(key,'[NaN]'))
            connection.close()
            with patch('requests.post',return_value=embeddings([[1,2]])) as post:
                self.assertEqual(client.embed(['document']),[[1.0,2.0]])
                self.assertEqual(post.call_count,1)

    def test_empty_input_and_invalid_roles_do_not_request_remote(self):
        with TemporaryDirectory() as directory,patch('requests.post') as post:
            client=SiliconFlowEmbedder(settings(),directory)
            self.assertEqual(client.embed([]),[])
            for value in ('text',[None],[' ']):
                with self.assertRaises(ProviderError):
                    client.embed(value)
            with self.assertRaises(ProviderError):
                client.embed(['text'],role='bad')
            with self.assertRaises(ProviderError):
                client.embed(['text'],role={})
            post.assert_not_called()


class HrChatAndErrorsTests(unittest.TestCase):
    def test_deepseek_separate_credentials_nonstream_and_no_answer_cache(self):
        chat=DeepSeekChat(settings())
        output={"model":"deepseek-flash","choices":[{"message":{"content":"答案 [F03]"}}],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14,"private":"do-not-export",DS_KEY:99}}
        with patch('requests.post',return_value=response(output)) as post:
            result=chat.chat([{'role':'user','content':'问题'}],max_tokens=1000)
            self.assertEqual(result,{'content':'答案 [F03]','usage':{'prompt_tokens':10,'completion_tokens':4,'total_tokens':14},'model':'deepseek-flash'})
            chat.chat([{'role':'user','content':'问题'}])
            self.assertEqual(post.call_count,2)
            self.assertEqual(post.call_args.args[0],'https://api.deepseek.com/chat/completions')
            body=post.call_args.kwargs['json']
            self.assertEqual(body['thinking'],{'type':'disabled'})
            self.assertFalse(body['stream'])
            self.assertEqual(body['temperature'],0)
            self.assertEqual(post.call_args.kwargs['headers']['Authorization'],'Bearer '+DS_KEY)
            self.assertNotIn(SF_KEY,repr(post.call_args))
            self.assertNotIn(DS_KEY,json.dumps(body))

    def test_redacts_known_secrets_in_returned_text_and_model(self):
        payload={'choices':[{'message':{'content':DS_KEY+' '+SF_KEY+' Bearer remote-token'}}],'model':SF_KEY}
        with patch('requests.post',return_value=response(payload)):
            result=DeepSeekChat(settings()).chat([{'role':'user','content':'test'}])
        serialized=json.dumps(result)
        for secret in (DS_KEY,SF_KEY,'remote-token'):
            self.assertNotIn(secret,serialized)

    def test_missing_keys_do_not_call_network(self):
        with TemporaryDirectory() as directory,patch('requests.post') as post:
            with self.assertRaises(ProviderError) as caught:
                DeepSeekChat(settings(deepseek_api_key='')).chat([{'role':'user','content':'q'}])
            self.assertEqual(caught.exception.code,'missing_key')
            with self.assertRaises(ProviderError):
                SiliconFlowEmbedder(settings(siliconflow_api_key=''),directory).embed(['text'])
            post.assert_not_called()

    def test_http_errors_exclude_remote_bodies_and_credentials(self):
        for status,code in [(401,'authentication'),(403,'permission'),(429,'rate_limit'),(500,'http_error'),(302,'http_error')]:
            with self.subTest(status=status):
                remote=response(status=status,text='remote body '+DS_KEY+' '+SF_KEY)
                with patch('requests.post',return_value=remote),self.assertRaises(ProviderError) as caught:
                    DeepSeekChat(settings()).chat([{'role':'user','content':'q'}])
                self.assertEqual(caught.exception.code,code)
                self.assertEqual(caught.exception.status,status)
                for secret in (DS_KEY,SF_KEY,'remote body'):
                    self.assertNotIn(secret,str(caught.exception))
                remote.json.assert_not_called()

    def test_timeout_and_connection_errors_exclude_original_exception(self):
        for error,code in [(requests.Timeout('secret '+DS_KEY),'timeout'),(requests.ConnectionError('Bearer '+SF_KEY),'connection')]:
            with self.subTest(code=code),patch('requests.post',side_effect=error),self.assertRaises(ProviderError) as caught:
                DeepSeekChat(settings()).chat([{'role':'user','content':'q'}])
            self.assertEqual(caught.exception.code,code)
            self.assertNotIn(DS_KEY,str(caught.exception))
            self.assertNotIn(SF_KEY,str(caught.exception))

    def test_invalid_json_body_and_reasoning_only_cannot_become_answers(self):
        bad=response()
        bad.json.side_effect=ValueError('private '+DS_KEY)
        cases=[bad,response([]),response({'error':{'message':DS_KEY}}),response({'choices':[]}),response({'choices':[{'message':{'content':'','reasoning_content':'thinking only'}}]})]
        for remote in cases:
            with self.subTest(response=remote),patch('requests.post',return_value=remote),self.assertRaises(ProviderError) as caught:
                DeepSeekChat(settings()).chat([{'role':'user','content':'q'}])
            self.assertEqual(caught.exception.code,'invalid_response')
            self.assertNotIn(DS_KEY,str(caught.exception))

    def test_invalid_chat_inputs_do_not_call_network(self):
        chat=DeepSeekChat(settings())
        with patch('requests.post') as post:
            for messages in ([],[{'role':'tool','content':'x'}],[{'role':[],'content':'x'}],[{'role':'user','content':''}]):
                with self.assertRaises(ProviderError):
                    chat.chat(messages)
            for max_tokens in (0,True,1.5):
                with self.assertRaises(ProviderError):
                    chat.chat([{'role':'user','content':'x'}],max_tokens=max_tokens)
            post.assert_not_called()


if __name__ == '__main__':
    unittest.main()
