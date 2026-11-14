/**
 * Behavioral checks for the strategy-specific parameter UI.
 * Run: node --test tests/hr_strategy_params_frontend.test.js
 * Uses the app's real inline JavaScript with a small DOM and job transport double.
 * It does not load a browser, contact model providers, or use saved credentials.
 */
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const sourcePath = path.resolve(__dirname, '../labs/hr_chunking_lab/web/index.html');

function loadLab() {
  const html = fs.readFileSync(sourcePath, 'utf8');
  const inline = html.match(/<script>([\s\S]*?)<\/script>/)[1];
  const bindingStart = inline.indexOf("document.querySelectorAll('[data-tab]')");
  assert.ok(bindingStart > 0, 'find declarations before browser bootstrap');
  const elements = new Map();
  const all = new Set();
  class Element {
    constructor(tag='div') {
      this.tagName = tag.toUpperCase(); this.children=[]; this.parentElement=null;
      this.dataset={}; this.attributes={}; this.listeners={}; this.value='';
      this.textContent=''; this.hidden=false; this.disabled=false; this.type='';
      this.style={setProperty(){}};
      const classes=new Set();
      this.classList={add:(...v)=>v.forEach(x=>classes.add(x)),remove:(...v)=>v.forEach(x=>classes.delete(x)),contains:v=>classes.has(v),toggle:(v,force)=>{const add=force===undefined?!classes.has(v):force;if(add)classes.add(v);else classes.delete(v);return add;}};
      all.add(this);
    }
    set id(v){if(this._id)elements.delete(this._id);this._id=v;if(v)elements.set(v,this);}
    get id(){return this._id||'';}
    set className(v){this._className=v;for(const word of v.split(/\s+/))if(word)this.classList.add(word);}
    get className(){return this._className||'';}
    append(...children){for(let child of children){if(typeof child==='string'){const n=new Element('#text');n.textContent=child;child=n;}child.parentElement=this;this.children.push(child);}}
    appendChild(child){this.append(child);return child;}
    replaceChildren(...children){for(const child of this.children)child.remove();this.children=[];this.append(...children);}
    remove(){for(const child of [...this.children])child.remove();if(this.id)elements.delete(this.id);all.delete(this);if(this.parentElement)this.parentElement.children=this.parentElement.children.filter(x=>x!==this);this.parentElement=null;}
    setAttribute(key,value){this.attributes[key]=String(value);if(key==='id')this.id=String(value);else if(key==='class')this.className=String(value);else if(key.startsWith('data-'))this.dataset[key.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())]=String(value);}
    getAttribute(key){return this.attributes[key]??null;}
    addEventListener(name,fn){(this.listeners[name]??=[]).push(fn);}
    dispatch(name){for(const fn of this.listeners[name]||[])fn({target:this,currentTarget:this});}
    checkValidity(){const n=Number(this.value);return this.type!=='number'||(this.value!==''&&Number.isFinite(n)&&(!this.min||n>=Number(this.min))&&(!this.max||n<=Number(this.max)));}
    focus(){};
    querySelectorAll(selector){return query(selector).filter(el=>{for(let p=el.parentElement;p;p=p.parentElement)if(p===this)return true;return false;});}
    querySelector(selector){return this.querySelectorAll(selector)[0]||null;}
    get previousElementSibling(){if(!this.parentElement)return {textContent:this.id};const i=this.parentElement.children.indexOf(this);return this.parentElement.children[i-1]||{textContent:this.id};}
  }
  function match(el,token){
    if(token.startsWith('#'))return el.id===token.slice(1);
    if(token.startsWith('.'))return el.classList.contains(token.slice(1));
    const attr=token.match(/^(\w+)?\[([^=\]]+)(?:=["']?([^"'\]]+)["']?)?\]$/);
    if(attr){const value=attr[2].startsWith('data-')?el.dataset[attr[2].slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())]:el.getAttribute(attr[2]);return(!attr[1]||el.tagName===attr[1].toUpperCase())&&value!==null&&value!==undefined&&(attr[3]===undefined||value===attr[3]);}
    return el.tagName===token.toUpperCase();
  }
  function query(selector){const found=[];for(const part of selector.split(',')){const tokens=part.trim().split(/\s+/);for(const el of all){if(!match(el,tokens.at(-1)))continue;let p=el.parentElement,good=true;for(let i=tokens.length-2;i>=0;i--){while(p&&!match(p,tokens[i]))p=p.parentElement;if(!p){good=false;break;}p=p.parentElement;}if(good&&!found.includes(el))found.push(el);}}return found;}
  // Static IDs let ordinary render code find its containers. Dynamic parameter
  // inputs are registered by the real renderStrategyParams implementation.
  for(const m of html.matchAll(/<([\w-]+)\b([^>]*\bid="([^"]+)"[^>]*)>/g)){
    const e=new Element(m[1]);e.id=m[3];
    for(const a of m[2].matchAll(/\b(value|min|max|step|type|class)="([^"]*)"/g)){if(a[1]==='class')e.className=a[2];else e[a[1]]=a[2];}
  }
  if(elements.get('retrievalMode')&&!elements.get('retrievalMode').value)elements.get('retrievalMode').value='vector';
  const body=new Element('body');
  const document={body,getElementById:id=>elements.get(id)||null,createElement:tag=>new Element(tag),createTextNode:text=>{const e=new Element('#text');e.textContent=text;return e;},querySelectorAll:query,querySelector:selector=>query(selector)[0]||null};
  const context=vm.createContext({document,console,Intl,Map,Set,setTimeout:()=>0,clearTimeout(){},fetch(){throw new Error('Unexpected network request in unit test');}});
  vm.runInContext(inline.slice(0,bindingStart)+`\n
    // Rendering the already tested result/source cards is outside these checks.
    renderRetrieval=()=>{};renderCompare=()=>{};renderIndexSummary=()=>{};updateControls=()=>{};toast=()=>{};
    globalThis.__calls=[];
    runJob=async(action,payload)=>{__calls.push({action,payload});return action==='retrieve'?{run_id:'test-run',question:payload.question,results:payload.strategies.map(strategy=>({strategy,candidates:[],stats:{chunk_count:1}}))}:{indexes:[]};};
    globalThis.__lab={state,METHOD_FIELDS,getSharedParams,getStrategyParams,getParams,validateParams,validateStrategies,signature,isStale,setActive,renderStrategyParams,retrieve,build,calls:__calls};
  `,context,{filename:sourcePath});
  const startupStart=inline.indexOf('    renderStrategies();renderIndexSummary();',bindingStart);
  assert.ok(startupStart>bindingStart,'find browser event bindings before initial render');
  vm.runInContext(inline.slice(bindingStart,startupStart),context,{filename:sourcePath});
  const lab=context.__lab;
  const plain=value=>JSON.parse(JSON.stringify(value));
  const input=(id,value)=>{const e=elements.get(id);assert.ok(e,`input ${id} exists`);e.value=String(value);e.dispatch('input');return e;};
  elements.get('questionInput').value='公司福利年假怎么申请？';
  return {lab,elements,plain,input,html};
}

function keys(value){return Object.keys(value).sort();}

test('each method exposes only its applicable chunk parameters',()=>{
  const {lab,plain}=loadLab();
  const map=plain(lab.getStrategyParams());
  for(const id of ['fixed','recursive','structure','contextual'])assert.deepEqual(keys(map[id]),['chunk_overlap','chunk_size']);
  assert.deepEqual(keys(map.semantic),['chunk_size','semantic_percentile']);
  assert.deepEqual(keys(map.parent_child),['chunk_overlap','chunk_size','parent_size']);
  assert.deepEqual(keys(plain(lab.getSharedParams())),['context_budget_chars','retrieval_mode','top_k']);
});

test('switching methods preserves separate values and removes unrelated controls',()=>{
  const {lab,elements,input}=loadLab();
  lab.setActive('fixed');input('chunkSize',420);input('chunkOverlap',20);
  lab.setActive('semantic');
  assert.equal(elements.has('chunkOverlap'),false,'semantic must not display overlap');
  assert.equal(elements.has('parentSize'),false,'semantic must not display parent size');
  input('chunkSize',180);input('semanticPercentile',90);
  lab.setActive('parent_child');input('chunkSize',200);input('parentSize',800);
  assert.equal(elements.has('semanticPercentile'),false,'parent-child must not display semantic percentile');
  lab.setActive('fixed');assert.equal(Number(elements.get('chunkSize').value),420);assert.equal(Number(elements.get('chunkOverlap').value),20);
  assert.equal(elements.has('parentSize'),false);
  lab.setActive('semantic');assert.equal(Number(elements.get('chunkSize').value),180);assert.equal(Number(elements.get('semanticPercentile').value),90);
});

test('unrelated hidden parent limit does not block large fixed chunks',()=>{
  const {lab,input}=loadLab();lab.setActive('fixed');input('chunkSize',1200);
  assert.equal(lab.validateParams('fixed').chunk_size,1200);
  assert.ok(lab.getParams('fixed').parent_size>=1200,'flat preview params remain accepted by the legacy validator');
});

test('invalid overlap is caught for its own method without blocking others',()=>{
  const {lab,input}=loadLab();lab.setActive('fixed');input('chunkSize',100);input('chunkOverlap',100);
  assert.throws(()=>lab.validateParams('fixed'),/重叠/);
  lab.setActive('semantic');assert.doesNotThrow(()=>lab.validateParams('semantic'));
  assert.throws(()=>lab.validateStrategies(['fixed','semantic']),/重叠/);
});

test('staleness follows each method snapshot, not the selected method',()=>{
  const {lab,input,elements}=loadLab();lab.setActive('fixed');
  lab.state.runSignatures.set('fixed',lab.signature('fixed'));lab.state.runSignatures.set('semantic',lab.signature('semantic'));
  lab.setActive('semantic');assert.equal(lab.isStale('fixed'),false);assert.equal(lab.isStale('semantic'),false);
  lab.setActive('fixed');input('chunkSize',420);
  assert.equal(lab.isStale('fixed'),true);assert.equal(lab.isStale('semantic'),false,'changing fixed must not stale semantic');
  elements.get('topK').value='5';assert.equal(lab.isStale('semantic'),true,'common retrieval setting must stale all existing runs');
});

test('build and comparison requests carry independent method settings',async()=>{
  const {lab,input,plain,elements}=loadLab();lab.setActive('fixed');input('chunkSize',400);input('chunkOverlap',30);
  lab.setActive('recursive');input('chunkSize',180);input('chunkOverlap',10);
  elements.get('retrievalMode').value='keyword';elements.get('topK').value='2';
  await lab.retrieve(['fixed','recursive']);
  const request=plain(lab.calls[0]);assert.equal(request.action,'retrieve');
  assert.deepEqual(request.payload.strategy_params,{fixed:{chunk_size:400,chunk_overlap:30},recursive:{chunk_size:180,chunk_overlap:10}});
  assert.deepEqual(request.payload.params,{retrieval_mode:'keyword',top_k:2,context_budget_chars:1800});
  await lab.build(['fixed','recursive']);
  const built=plain(lab.calls[1]);assert.equal(built.action,'build');assert.deepEqual(built.payload.strategy_params,request.payload.strategy_params);
});


test('editing one method invalidates only its index, while retrieval controls have their proper scope',()=>{
  const {lab,input}=loadLab();lab.setActive('fixed');
  for(const id of ['fixed','recursive','semantic'])lab.state.indexes.set(id,{status:'ready'});
  input('chunkSize',420);
  assert.equal(lab.state.indexes.has('fixed'),false);
  assert.equal(lab.state.indexes.has('recursive'),true);
  assert.equal(lab.state.indexes.has('semantic'),true);
  input('topK',5);input('contextBudget',2200);
  assert.equal(lab.state.indexes.size,2,'Top K and material budget do not change built chunks');
  input('retrievalMode','keyword');
  assert.equal(lab.state.indexes.size,0,'changing retrieval/index mode invalidates all index indicators');
});

test("resetting public settings preserves each method's custom parameters",()=>{
  const {lab,input,elements,plain}=loadLab();
  lab.setActive('fixed');input('chunkSize',420);input('chunkOverlap',20);
  lab.setActive('semantic');input('chunkSize',180);input('semanticPercentile',90);
  input('topK',5);input('contextBudget',2200);input('retrievalMode','keyword');
  const before=plain(lab.getStrategyParams());
  elements.get('resetParams').dispatch('click');
  assert.deepEqual(plain(lab.getStrategyParams()),before);
  assert.deepEqual(plain(lab.getSharedParams()),{top_k:3,context_budget_chars:1800,retrieval_mode:'vector'});
});

test('resetting one method leaves other methods and public settings intact',()=>{
  const {lab,input,elements,plain}=loadLab();
  lab.setActive('fixed');input('chunkSize',420);input('chunkOverlap',20);
  lab.setActive('semantic');input('chunkSize',180);input('semanticPercentile',90);
  input('topK',5);input('contextBudget',2200);input('retrievalMode','keyword');
  for(const id of ['fixed','recursive','semantic'])lab.state.indexes.set(id,{status:'ready'});
  const shared=plain(lab.getSharedParams());
  elements.get('resetStrategyParams').dispatch('click');
  assert.deepEqual(plain(lab.getStrategyParams(['semantic'])),{semantic:{chunk_size:280,semantic_percentile:75}});
  assert.deepEqual(plain(lab.getStrategyParams(['fixed'])),{fixed:{chunk_size:420,chunk_overlap:20}});
  assert.deepEqual(plain(lab.getSharedParams()),shared);
  assert.equal(lab.state.indexes.has('semantic'),false);
  assert.equal(lab.state.indexes.has('fixed'),true);
  assert.equal(lab.state.indexes.has('recursive'),true);
});
