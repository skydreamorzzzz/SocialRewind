"""离线、单文件审计页；不启动服务，不调用模型，也不是 Agent 观察接口。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_timeline_page(output: dict[str, Any], path: Path) -> None:
    # 页面只需要事件轨迹；完整请求和响应留在 JSON，避免嵌入重复对话导致页面膨胀。
    data = {"mode": output["mode"], "status": output["status"], "summary": output["summary"],
            "experiments": []}
    for experiment in output["experiments"]:
        runs = []
        for run in experiment["runs"]:
            events = experiment["worlds"][run["group"]]["events"][:run["audit_end"]]
            runs.append({"group": run["group"], "life": run["life"], "status": run["status"],
                         "metrics": run.get("metrics"), "error": run.get("error"),
                         "memory_count": {r: 1 + len(m["previous_lives"]) for r, m in run["memories"].items()},
                         "events": [e for e in events if e["timeline_id"] == run["life"] and e["kind"] != "reset"]})
        data["experiments"].append({"id": experiment["id"], "scenario": experiment["scenario"], "runs": runs})
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False)
    for char, replacement in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                              ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        payload = payload.replace(char, replacement)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_PAGE.replace("__EXPERIMENT_DATA__", payload), encoding="utf-8")


_PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SocialRewind · 时间线实验室</title>
<style>
:root{font:15px/1.65 system-ui,sans-serif;color:#e5edf7;background:#0c1423;color-scheme:dark}
*{box-sizing:border-box}body{margin:0}main{max-width:1500px;margin:auto;padding:32px 24px}
h1{font-size:32px;letter-spacing:-1px;margin:0}h2{font-size:21px}h3{margin:0;font-size:17px}
p{margin:8px 0 16px}.muted{color:#a7b9d0}.tag{font-size:12px;letter-spacing:2px;color:#7fe1cd}
.toolbar{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin:24px 0}
select{max-width:100%;padding:10px;border:1px solid #455672;border-radius:8px;background:#172337;color:inherit}
label{cursor:pointer}.notice{border-left:3px solid #7fe1cd;padding:8px 16px;background:#132436}
.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}
td,th{text-align:left;padding:10px 14px;border-bottom:1px solid #293b55;white-space:nowrap}th{color:#9edfd4}
.columns{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,350px),1fr));gap:18px;align-items:start}
.timeline{border:1px solid #31445f;border-radius:14px;overflow:hidden;background:#111e30}
.head{padding:18px;border-top:3px solid #719cdb}.rewind .head{border-color:#7fe1cd}.control .head{border-color:#d5ae75}
.result{font-variant-numeric:tabular-nums;padding:12px 0;color:#d9e6f7;white-space:pre-line}.change{font-size:13px;color:#a7b9d0}
.chat{max-height:670px;overflow:auto;padding:0 14px 14px}.event{padding:14px;margin:10px 0;border-radius:10px;background:#1a2c43;border-left:3px solid #78a8ed}
.seller{background:#1a343b;border-color:#7fe1cd}.refused{border-color:#f18e99}.event small{color:#a7b9d0}
.speech{white-space:pre-wrap;overflow-wrap:anywhere;margin:6px 0}.terms{font-family:ui-monospace,monospace;font-size:13px;color:#e4cc9d}
details{font-size:13px;margin-top:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}
footer{margin-top:30px;color:#a7b9d0;font-size:13px}input:focus-visible,select:focus-visible,summary:focus-visible{outline:2px solid #7fe1cd}
</style></head><body><main>
<div class="tag">THE WORLD RESETS. EXPERIENCE REMAINS.</div>
<h1>SocialRewind <span class="muted">/ 时间线实验室</span></h1>
<p class="muted">同一初始世界，私人偏好，三维合同。并列查看自由谈判与各世结果。</p>
<p class="notice">实验审计视图：可包含双方私人决策，仅供研究者本地使用。模型完整请求与响应位于同名 JSON 日志。</p>
<div class="toolbar"><label>实验 <select id="experiment" aria-label="选择偏好配置与重复实验"></select></label>
<label><input id="notes" type="checkbox"> 展开显式决策说明（审计权限）</label></div>
<p id="mode" class="muted"></p><div id="summary" class="scroll"></div>
<div id="timelines" class="columns"></div>
<details><summary>本配置的私人偏好与客观效用公式（审计）</summary><pre id="scenario"></pre></details>
<footer>条款变化由记录逐项比较；决策说明是 Agent 自述，不是策略学习的自动判定。Control 不保留前世经历，Rewind 只携带各自授权历史。记忆及叙事情境同时变化，单次结果不能证明因果改善。</footer>
</main><script id="data" type="application/json">__EXPERIMENT_DATA__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('data').textContent);
const picker=document.getElementById('experiment'), notes=document.getElementById('notes');
const names={baseline:'第一世',control:'Control',rewind:'Rewind'};
function el(tag,text,cls){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(cls)node.className=cls;return node;}
function terms(t){return t?`${t.price} 元 · ${t.delivery_days} 天交付 · ${t.warranty_days} 天保障`:'未成交';}
function firstOffer(run){return run.events.find(e=>e.accepted&&e.kind==='offer');}
function draw(){
 const exp=data.experiments[Number(picker.value)]; if(!exp)return;
 document.getElementById('mode').textContent=`模式：${data.mode} / 状态：${data.status} / ${exp.id}`;
 document.getElementById('scenario').textContent=JSON.stringify(exp.scenario,null,2);
 const summary=document.getElementById('summary');summary.replaceChildren();
 const table=el('table'),head=el('tr');['时间线','结果','合同','Buyer Utility','Seller Utility','Social Welfare','Pareto','Rounds'].forEach(t=>head.append(el('th',t)));table.append(head);
 const columns=document.getElementById('timelines');columns.replaceChildren();
 for(const run of exp.runs){
  const m=run.metrics,tr=el('tr');
  [names[run.group]+' / '+run.life, m?m.status:run.status,m?terms(m.deal_terms):'评价未完成',m?.buyer_utility??'—',m?.seller_utility??'—',m?.social_welfare??'—',m?(m.pareto_efficient?'是':'否'):'—',m?.rounds??'—'].forEach(v=>tr.append(el('td',String(v))));table.append(tr);
  const card=el('section',undefined,'timeline '+run.group), top=el('div',undefined,'head');
  top.append(el('h2',`${names[run.group]} / 第 ${run.life} 世`));
  top.append(el('div',Object.keys(run.memory_count).length?`授权记忆：Buyer ${run.memory_count.buyer} 世，Seller ${run.memory_count.seller} 世`:'无前世记忆','muted'));
  top.append(el('div',m?`${terms(m.deal_terms)}\n收益 ${m.buyer_utility} / ${m.seller_utility} · 福利 ${m.social_welfare}`:JSON.stringify(run.error),'result'));
  const offer=firstOffer(run), prior=exp.runs.filter(r=>(r.group===run.group||r.group==='baseline')&&r.life<run.life).at(-1);
  if(offer){top.append(el('div','首个正式方案：'+terms(offer.input),'change'));
   if(prior&&firstOffer(prior)){const old=firstOffer(prior).input;const changes=['price','delivery_days','warranty_days'].filter(k=>old[k]!==offer.input[k]).map(k=>`${k}: ${old[k]} → ${offer.input[k]}`);top.append(el('div','相较前世首案：'+(changes.join('；')||'条款相同'),'change'));}
  }
  card.append(top);const chat=el('div',undefined,'chat');
  for(const event of run.events){
   const item=el('article',undefined,'event '+event.actor+(event.accepted?'':' refused'));
   item.append(el('small',`#${event.timeline_sequence} · ${event.actor} · ${event.kind}${event.accepted?'':' · 被拒绝'}`));
   item.append(el('p',event.input.text??'', 'speech'));
   if(event.kind==='offer')item.append(el('div',terms(event.input),'terms'));
   if(event.input.offer_id)item.append(el('div','回应 '+event.input.offer_id,'terms'));
   if(!event.accepted)item.append(el('pre',JSON.stringify(event.result)));
   if(notes.checked){const detail=el('details');detail.open=true;detail.append(el('summary','本人显式决策说明（未经事实核查）'));detail.append(el('pre',JSON.stringify(event.decision,null,2)));item.append(detail);}
   chat.append(item);
  }
  card.append(chat);columns.append(card);
 }
 summary.append(table);
}
data.experiments.forEach((e,i)=>{const option=el('option',e.id);option.value=String(i);picker.append(option);});
picker.addEventListener('change',draw);notes.addEventListener('change',draw);draw();
</script></body></html>'''
