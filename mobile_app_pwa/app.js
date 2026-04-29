const $ = (id) => document.getElementById(id);
const apiBaseEl = $('apiBase');

function defaultApiBase(){
  const qp = new URLSearchParams(window.location.search).get('api');
  if (qp) return qp;
  const host = window.location.hostname;
  if (host === '127.0.0.1' || host === 'localhost') return 'http://127.0.0.1:8000';
  return window.location.origin;
}

apiBaseEl.value = defaultApiBase();

function api(path){return `${apiBaseEl.value.replace(/\/$/,'')}${path}`;}

async function checkHealth(){
  const out=$('healthText');
  try{
    const r=await fetch(api('/health'));
    const j=await r.json();
    out.textContent=`API ok: ${JSON.stringify(j)}`;
  }catch(e){out.textContent=`API error: ${e}`;}
}

async function getRecs(){
  const userId=Number($('userId').value);
  const topK=Number($('topK').value);
  const includeSeen=$('includeSeen').checked;
  const tbody=document.querySelector('#recTable tbody');
  tbody.innerHTML='';
  try{
    const r=await fetch(api(`/recommend/${userId}?top_k=${topK}&include_seen=${includeSeen}`));
    const j=await r.json();
    (j.items||[]).forEach(it=>{
      const tr=document.createElement('tr');
      tr.innerHTML=`<td>${it.rank}</td><td>${it.item_id}</td><td>${Number(it.score).toFixed(4)}</td><td>${it.reason||''}</td>`;
      tbody.appendChild(tr);
    });
  }catch(e){
    const tr=document.createElement('tr');
    tr.innerHTML=`<td colspan="4">Error: ${e}</td>`;
    tbody.appendChild(tr);
  }
}

async function submitInteraction(){
  const out=$('interactText');
  const body={
    user_id:Number($('userId').value),
    item_id:Number($('itemId').value),
    behavior:Number($('behavior').value)
  };
  try{
    const r=await fetch(api('/interact'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j=await r.json();
    out.textContent=`${j.message||'ok'}; recommendations refreshed`;
    await getRecs();
  }catch(e){out.textContent=`submit error: ${e}`;}
}

$('healthBtn').onclick=checkHealth;
$('recommendBtn').onclick=getRecs;
$('interactBtn').onclick=submitInteraction;

if ('serviceWorker' in navigator){ navigator.serviceWorker.register('sw.js').catch(()=>{}); }
