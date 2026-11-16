"""离线、单文件审计页；不启动服务，不调用模型，也不是 Agent 观察接口。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_timeline_page(output: dict[str, Any], path: Path) -> None:
    # 页面只需要事件轨迹；完整请求和响应留在 JSON，避免嵌入重复对话导致页面膨胀。
    data = {"mode": output["mode"], "status": output["status"], "summary": output["summary"],
            "design": output.get("design"), "ablation_statistics": output.get("ablation_statistics"),
            "counts": output.get("counts"), "experiments": []}
    for experiment in output["experiments"]:
        runs = []
        for run in experiment["runs"]:
            events = experiment["worlds"][run["group"]]["events"][:run["audit_end"]]
            runs.append({"group": run["group"], "life": run["life"], "status": run["status"],
                         "metrics": run.get("metrics"), "error": run.get("error"),
                         "memories": run["memories"], "reflections": run.get("reflections", {}),
                         "estimations": run.get("estimations", {}),
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
.rewind_hr .head{border-color:#7fe1cd}.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:16px;margin:24px 0}
.chart{padding:16px;background:#142337;border:1px solid #31445f;border-radius:12px}.bar-row{margin:12px 0}.bar{height:10px;background:#719cdb;border-radius:4px}.control .bar{background:#d5ae75}.rewind_hr .bar{background:#7fe1cd}.bar-label{font-size:13px;color:#a7b9d0}
.result{font-variant-numeric:tabular-nums;padding:12px 0;color:#d9e6f7;white-space:pre-line}.change{font-size:13px;color:#a7b9d0}
.chat{max-height:570px;overflow:auto;padding:0 14px 14px}.event{padding:14px;margin:10px 0;border-radius:10px;background:#1a2c43;border-left:3px solid #78a8ed}
.compact .chat{max-height:230px}.compact{margin:16px 0 24px}.memory,.strategy,.outcome{padding:16px 18px;border-top:1px solid #31445f}
.memory{background:#142d36}.memory p,.strategy p{margin:6px 0;font-size:14px}.memory details{margin:8px 0}
.personal{padding:8px 0}.personal strong{color:#9edfd4}.outcome{background:#1a2d41}.step{font-size:12px;letter-spacing:1px;color:#7fe1cd}
#first-life>summary{font-size:18px;cursor:pointer}.background{max-width:1000px}.empty{padding:18px;color:#a7b9d0}
.seller{background:#1a343b;border-color:#7fe1cd}.refused{border-color:#f18e99}.event small{color:#a7b9d0}
.speech{white-space:pre-wrap;overflow-wrap:anywhere;margin:6px 0}.terms{font-family:ui-monospace,monospace;font-size:13px;color:#e4cc9d}
details{font-size:13px;margin-top:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}
footer{margin-top:30px;color:#a7b9d0;font-size:13px}input:focus-visible,select:focus-visible,summary:focus-visible{outline:2px solid #7fe1cd}
</style></head><body><main>
<div class="tag">THE WORLD RESETS. EXPERIENCE REMAINS.</div>
<h1>SocialRewind <span class="muted">/ 时间线实验室</span></h1>
<p class="muted background" id="background"></p>
<p class="notice">前世谈判 → 保留的个人经验 → 本世选择 → 合同与收益。虚构案例，运行模式见下方标记；此页为研究者审计视图，含双方私人记录。完整模型调用保存在实验 JSON。</p>
<p id="sample-count" class="muted"></p>
<div id="charts" class="charts"></div>
<details id="statistics-panel" hidden><summary>各组统计、配对差值与95%置信区间</summary><p>误差分别对应买方估计卖方最低合法价、卖方估计买方最高合法价。固定3天/30天；不是理想成交价或下一轮报价。配对 bootstrap 区间以独立重复为单位，小样本仅供探索。</p><pre id="statistics"></pre></details>
<div class="toolbar"><label>实验 <select id="experiment" aria-label="选择偏好配置与重复实验"></select></label>
<label>对比第 <select id="life" aria-label="选择对比世数"></select> 世</label>
<label><input id="notes" type="checkbox"> 展开显式决策说明（审计权限）</label></div>
<p id="mode" class="muted"></p><div id="summary" class="scroll"></div>
<details id="first-life" open><summary>共同第一世：所有分支从同一场谈判重启</summary><div id="baseline"></div></details>
<div id="timelines" class="columns"></div>
<details><summary>本配置的私人偏好与客观效用公式（审计）</summary><pre id="scenario"></pre></details>
<footer id="research-note">条款变化由记录逐项比较；决策说明是 Agent 自述，不是策略学习的自动判定。Control 不保留前世经历，Rewind 只携带各自授权历史。记忆及叙事情境同时变化，单次结果不能证明因果改善。</footer>
</main><script id="data" type="application/json">__EXPERIMENT_DATA__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('data').textContent);
const picker=document.getElementById('experiment'), lifePicker=document.getElementById('life'), notes=document.getElementById('notes');
const names={baseline:'第一世',control:'Control',rewind:'Rewind',rewind_h:'Rewind-H',rewind_hr:'Rewind-HR'};
function el(tag,text,cls){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(cls)node.className=cls;return node;}
function terms(t){return t?`${t.price} 元 · ${t.delivery_days} 天交付 · ${t.warranty_days} 天保障`:'未成交';}
function firstOffer(run,role){return run.events.find(e=>e.accepted&&e.kind==='offer'&&e.actor===role);}
function personal(summary){
 const box=el('div',undefined,'personal');
 if(!summary){box.append(el('p','此历史记录未生成个人总结。','muted'));return box;}
 box.append(el('strong',`${summary.role} · 第 ${summary.source_timeline_id} 世${summary.source==='offline_fixture'?'离线机制占位总结':'的个人自述'}`));
 for(const [label,text] of [['观察',summary.observations.join('；')],['对对方的判断（可能错误）',summary.beliefs],['下一世可能尝试',summary.next_strategy]])box.append(el('p',label+'：'+text));
 return box;
}
function cardFor(run,exp,compact=false){
 const card=el('section',undefined,'timeline '+run.group+(compact?' compact':'')),top=el('div',undefined,'head');
 top.append(el('h2',`${names[run.group]} / 第 ${run.life} 世`));
 top.append(el('div',run.group==='rewind_h'?'双方仅携带本人完整授权历史':(run.group==='rewind'||run.group==='rewind_hr')?'双方携带本人授权历史与 Reflection':'无可用前世历史或 Reflection','muted'));card.append(top);
 if(!compact){
  const memory=el('div',undefined,'memory');memory.append(el('div','01 / 本世保留的经验','step'));
  if(run.group==='control')memory.append(el('p','双方知道世界发生过重置，但没有具体前世经历。下方本世结束后的总结仅保存给审计者，下一世不注入。'));
  for(const [role,m] of Object.entries(run.memories)){
   memory.append(el('p',`${role} 记得第 ${m.source_timeline_id} 世结果：${terms(m.outcome.deal_terms)}，本人收益 ${m.outcome.own_utility}。`));
   if(run.group==='rewind_h')memory.append(el('p','本组不注入 Reflection；完整授权历史仍然保留。'));
   else memory.append(personal(m.personal_summary));
   const history=el('details');history.append(el('summary',`查看 ${role} 继承的 ${m.visible_events.length} 条授权事件`));history.append(el('pre',JSON.stringify(m.visible_events,null,2)));memory.append(history);
   if(m.previous_lives.length){const earlier=el('details');earlier.append(el('summary','更早的本人经验'));m.previous_lives.forEach(old=>earlier.append(personal(old.personal_summary)));memory.append(earlier);}
  }
  card.append(memory);
  const strategy=el('div',undefined,'strategy');strategy.append(el('div','02 / 本世实际开场与报价变化','step'));
  const prior=exp.runs.filter(r=>(r.group===run.group||r.group==='baseline')&&r.life<run.life).at(-1);
  for(const role of ['buyer','seller']){
   const first=run.events.find(e=>e.actor===role),offer=firstOffer(run,role),old=prior&&firstOffer(prior,role);
   if(first)strategy.append(el('p',`${role} 首行动 ${first.kind}；本人说明：${first.decision.intent}`));
   if(offer){strategy.append(el('p',`${role} 首案：${terms(offer.input)}`,'change'));
    if(old){const changes=['price','delivery_days','warranty_days'].filter(k=>old.input[k]!==offer.input[k]).map(k=>`${k}: ${old.input[k]} → ${offer.input[k]}`);strategy.append(el('p','研究者相较前世首案：'+(changes.join('；')||'相同'),'change'));}}
  }
  strategy.append(el('p','以上是行动与角色自述的逐项对照，不自动认定经验导致了策略改变。','muted'));card.append(strategy);
 }
 const chat=el('div',undefined,'chat');chat.append(el('div',compact?'共同前世的公开互动':'03 / 本世公开互动','step'));
 for(const event of run.events){
  const item=el('article',undefined,'event '+event.actor+(event.accepted?'':' refused'));
  item.append(el('small',`#${event.timeline_sequence} · ${event.actor} · ${event.kind}${event.accepted?'':' · 被拒绝'}`));
  item.append(el('p',event.input.text??'','speech'));
  if(event.kind==='offer')item.append(el('div',`${event.result.offer_id??''} · ${terms(event.input)}`,'terms'));
  if(event.input.offer_id)item.append(el('div','回应 '+event.input.offer_id,'terms'));
  if(!event.accepted)item.append(el('pre',JSON.stringify(event.result)));
  if(notes.checked){const detail=el('details');detail.open=true;detail.append(el('summary','本人显式决策说明（未经事实核查）'));detail.append(el('pre',JSON.stringify(event.decision,null,2)));item.append(detail);}
  chat.append(item);
 }
 card.append(chat);const outcome=el('div',undefined,'outcome'),m=run.metrics;
 outcome.append(el('div','04 / 最终合同与收益','step'));
 outcome.append(el('p',m?`${m.status} · ${terms(m.deal_terms)}\nBuyer ${m.buyer_utility} / Seller ${m.seller_utility} / Welfare ${m.social_welfare} / ${m.rounds} 轮 / Pareto ${m.pareto_efficient?'是':'否'}`:'尚无完整谈判结果','result'));
 if(run.error)outcome.append(el('pre','运行失败：'+JSON.stringify(run.error)));
 if(Object.keys(run.estimations).length){
  const estimation=el('div',undefined,'estimation');estimation.append(el('h3','事后客观接受边界估计 · 3天/30天'));
  estimation.append(el('p','仅审计展示：估计、真值和误差从不传入后续记忆。','muted'));
  for(const [role,item] of Object.entries(run.estimations)){
   estimation.append(el('p',`${role} 估计 ${item.target_role} 的${role==='buyer'?'最低':'最高'}合法价：${item.price} 元；真值 ${item.true_boundary}；绝对误差 ${item.absolute_error} 元。`));
   const detail=el('details');detail.append(el('summary','估计依据与不确定性'));detail.append(el('p',item.reason));detail.append(el('p',item.uncertainty));estimation.append(detail);
  }outcome.append(estimation);
 }
 const generated=el('details');generated.append(el('summary',run.group==='control'?'本世个人总结（仅审计保存，不传给下一世）':'本世个人总结（本世结束后生成，供后续 Rewind 继承）'));
 Object.values(run.reflections).forEach(s=>generated.append(personal(s)));outcome.append(generated);
 const audit=el('details');audit.append(el('summary','查看原始审计记录'));audit.append(el('pre',JSON.stringify(run,null,2)));outcome.append(audit);card.append(outcome);return card;
}
function setLives(){
 const exp=data.experiments[Number(picker.value)];lifePicker.replaceChildren();
 const lives=[...new Set(exp.runs.filter(r=>r.group!=='baseline').map(r=>r.life))];
 (lives.length?lives:[1]).forEach(n=>{const option=el('option',String(n));option.value=String(n);lifePicker.append(option);});
}
function draw(){
 const exp=data.experiments[Number(picker.value)]; if(!exp)return;
 document.getElementById('mode').textContent=`模式：${data.mode} / 状态：${data.status} / ${exp.id}`;
 document.getElementById('scenario').textContent=JSON.stringify(exp.scenario,null,2);
 document.getElementById('background').textContent=`${exp.scenario.product.name} · ${exp.scenario.product.condition} · ${exp.scenario.product.included} · 挂牌 ${exp.scenario.listed_price} 元。可协商价格、交付和保障。`;
 const summary=document.getElementById('summary');summary.replaceChildren();
 const table=el('table'),head=el('tr');const headers=['时间线','结果','合同','Buyer Utility','Seller Utility','Social Welfare','Pareto','Rounds'];if(data.design==='memory_ablation')headers.push('买方估计误差','卖方估计误差');headers.forEach(t=>head.append(el('th',t)));table.append(head);
 const columns=document.getElementById('timelines');columns.replaceChildren();
 const baseline=document.getElementById('baseline');baseline.replaceChildren();
 for(const run of exp.runs){
  const m=run.metrics,tr=el('tr');
  const cells=[names[run.group]+' / '+run.life,run.status==='failed'?'运行失败':m?m.status:run.status,m?terms(m.deal_terms):'评价未完成',m?.buyer_utility??'—',m?.seller_utility??'—',m?.social_welfare??'—',m?(m.pareto_efficient?'是':'否'):'—',m?.rounds??'—'];if(data.design==='memory_ablation')cells.push(run.estimations.buyer?.absolute_error??'—',run.estimations.seller?.absolute_error??'—');cells.forEach(v=>tr.append(el('td',String(v))));table.append(tr);
  if(run.group==='baseline')baseline.append(cardFor(run,exp,true));
  else if(run.life===Number(lifePicker.value))columns.append(cardFor(run,exp));
 }
 summary.append(table);
}
function drawStatistics(){
 if(data.design!=='memory_ablation')return;
 if(data.counts)document.getElementById('sample-count').textContent=`独立重复：尝试 ${data.counts.attempted_repeats}/${data.counts.requested_repeats}，完整配对 ${data.counts.complete_repeats}；场次：完整 ${data.counts.completed_runs}/${data.counts.attempted_runs}，失败 ${data.counts.failed_runs}；模型调用 ${data.counts.model_calls}。图表按各指标有效 n 汇总，失败与未发生的分支不伪造结果。`;
 document.getElementById('research-note').textContent='Control、Rewind-H、Rewind-HR 共享重启叙事、案例和基础谈判 Prompt；差异仅为记忆可用性。置信区间是小样本探索，未作多重比较校正。经验可被使用，也可能造成锚定，不能仅凭角色自述断言因果。';
 const panel=document.getElementById('statistics-panel');panel.hidden=false;
 document.getElementById('statistics').textContent=JSON.stringify({counts:data.counts,statistics:data.ablation_statistics},null,2);
 if(!data.ablation_statistics)return;
 const groups=['control','rewind_h','rewind_hr'];
 for(const [key,title] of [['buyer_utility','买方收益'],['seller_utility','卖方收益'],['buyer_estimation_error','买方估计卖方边界：误差 ↓'],['seller_estimation_error','卖方估计买方边界：误差 ↓']]){
  const chart=el('section',undefined,'chart');chart.append(el('h3',title));
  const max=Math.max(1,...groups.map(g=>data.ablation_statistics.groups[g][key].mean??0));
  for(const group of groups){const stats=data.ablation_statistics.groups[group][key],row=el('div',undefined,'bar-row '+group);
   row.append(el('div',`${names[group]} · ${stats.mean===null?'无数据':stats.mean.toFixed(1)} ± ${stats.sample_std===null?'不可估计':stats.sample_std.toFixed(1)} · n=${stats.n}`,'bar-label'));
   const bar=el('div',undefined,'bar');bar.style.width=`${100*(stats.mean??0)/max}%`;row.append(bar);chart.append(row);}
  chart.append(el('small','均值 ± 样本标准差；单位：元','muted'));document.getElementById('charts').append(chart);
 }
}
data.experiments.forEach((e,i)=>{const option=el('option',e.id);option.value=String(i);picker.append(option);});
picker.addEventListener('change',()=>{setLives();draw();});lifePicker.addEventListener('change',draw);notes.addEventListener('change',draw);drawStatistics();setLives();draw();
</script></body></html>'''
