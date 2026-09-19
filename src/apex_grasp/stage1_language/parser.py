from __future__ import annotations
import importlib, json, os, re
from dataclasses import dataclass
from typing import Protocol, Any
from urllib import request
from apex_grasp.common.config import require
from apex_grasp.common.errors import PipelineFailure
from apex_grasp.common.models import TaskIR, TargetDescription, ReferenceDescription, Relation, Constraints, Provenance, Attributes

class ParserAdapter(Protocol):
    def parse(self, text: str) -> dict[str, Any]: ...

@dataclass
class RuleParser:
    """Deterministic sanity/test parser. It is intentionally narrow and not an optimum benchmark parser."""
    parser_version: str
    ontology: dict[str, Any]

    def parse(self, text: str) -> dict[str, Any]:
        original=text
        t=" ".join(text.strip().split())
        low=t.lower()
        action="UNKNOWN"
        for surface, canon in [("pick up","GRASP"),("grab","GRASP"),("pick","GRASP"),("lift","GRASP"),("grasp","GRASP"),("place","PLACE"),("move","MOVE"),("inspect","INSPECT")]:
            if re.search(r"\b"+re.escape(surface)+r"\b",low): action=canon; break
        known_rel=[("wedged between","WEDGED_BETWEEN"),("to the left of","LEFT_OF"),("left of","LEFT_OF"),("to the right of","RIGHT_OF"),("right of","RIGHT_OF"),("in front of","IN_FRONT_OF"),("behind","BEHIND"),("above","ABOVE"),("over","ABOVE"),("below","BELOW"),("underneath","BELOW"),("between","BETWEEN"),("near","NEAR"),("inside","INSIDE")]
        rel_hit=next(((s,c,low.find(s)) for s,c in known_rel if s in low),None)
        unresolved=[]; ambiguity=0.05
        if re.search(r"\b(it|one|that one|this one)\b", low): unresolved.append("unresolved_pronoun"); ambiguity=max(ambiguity,0.6)
        if action=="UNKNOWN": unresolved.append("unsupported_action"); ambiguity=max(ambiguity,0.8)
        # Minimal noun extraction from ontology/object vocabulary is deliberately conservative.
        candidate_nouns=[]
        for noun in self.ontology.get("objects",[]):
            for m in re.finditer(r"\b"+re.escape(noun)+r"\b",low): candidate_nouns.append((m.start(),noun))
        candidate_nouns.sort()
        target_cat=candidate_nouns[0][1] if candidate_nouns else None
        if target_cat is None: unresolved.append("target_category_unknown"); ambiguity=max(ambiguity,0.5)
        colors=set(self.ontology.get("attributes",{}).get("color",[])); conditions=set(self.ontology.get("attributes",{}).get("condition",[])); sizes=set(self.ontology.get("attributes",{}).get("size",[])); trans=set(self.ontology.get("attributes",{}).get("transparency",[]))
        def nearest_before(words, pos):
            found=None
            for w in words:
                p=low.rfind(w,0,pos)
                if p>=0 and (found is None or p>found[0]): found=(p,w)
            return found[1] if found and pos-found[0] < 30 else None
        target_pos=candidate_nouns[0][0] if candidate_nouns else len(low)
        attrs=Attributes(color=nearest_before(colors,target_pos),condition=nearest_before(conditions,target_pos),size=nearest_before(sizes,target_pos))
        refs=[]; relations=[]
        if rel_hit:
            surface,canon,pos=rel_hit
            after=[x for x in candidate_nouns if x[0]>pos+len(surface)]
            if canon=="BETWEEN" or canon=="WEDGED_BETWEEN":
                after=after[:2]
                if len(after)<2: unresolved.append("between_requires_two_references"); ambiguity=max(ambiguity,0.6)
            else: after=after[:1]
            ids=[]
            for i,(p,noun) in enumerate(after,1):
                rid=f"ref_{i}"; ids.append(rid)
                refs.append(ReferenceDescription(id=rid,category=noun,attributes=Attributes(transparency=nearest_before(trans,p))))
            if ids: relations.append(Relation(type=canon,target="TARGET",reference=ids if len(ids)>1 else ids[0]))
        pref="exposed_side" if "exposed side" in low else ("top" if "use the top" in low else None)
        return TaskIR(action=action,target=TargetDescription(category=target_cat,attributes=attrs),references=refs,relations=relations,constraints=Constraints(preferred_grasp_region=pref),ambiguity=ambiguity,unresolved=unresolved,provenance=Provenance(parser_version=self.parser_version,original_instruction=original,model="rule")).model_dump()

@dataclass
class OpenAICompatibleParser:
    base_url: str; model: str; api_key_env: str; timeout_s: float; parser_version: str; ontology: dict[str,Any]
    def parse(self,text:str)->dict[str,Any]:
        key=os.getenv(self.api_key_env)
        if not key: raise PipelineFailure("CONFIG_REQUIRED",f"Environment variable {self.api_key_env} is not set")
        schema=TaskIR.model_json_schema()
        prompt=("Return ONLY JSON matching the supplied TaskIR JSON Schema. Do not invent unsupported details. Unknown values must be null; unresolved pronouns must be listed and ambiguity > 0. The parser must not inspect any image.\n"
                f"ONTOLOGY={json.dumps(self.ontology)}\nSCHEMA={json.dumps(schema)}\nINSTRUCTION={text}")
        payload={"model":self.model,"messages":[{"role":"user","content":prompt}],"temperature":0}
        req=request.Request(self.base_url.rstrip("/")+"/chat/completions",data=json.dumps(payload).encode(),headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"})
        try:
            with request.urlopen(req,timeout=self.timeout_s) as r: data=json.loads(r.read())
        except Exception as e: raise PipelineFailure("LANGUAGE_PARSE_FAILED",f"Parser endpoint request failed: {e}")
        content=data["choices"][0]["message"]["content"].strip()
        if content.startswith("```"): content=re.sub(r"^```(?:json)?\s*|\s*```$","",content,flags=re.S)
        try: obj=json.loads(content)
        except Exception as e: raise PipelineFailure("LANGUAGE_PARSE_FAILED",f"Model did not return valid JSON: {e}")
        obj.setdefault("provenance",{}); obj["provenance"].update({"parser_version":self.parser_version,"original_instruction":text,"model":self.model})
        return obj

@dataclass
class PythonCallableParser:
    module:str; function:str
    def parse(self,text:str)->dict[str,Any]:
        fn=getattr(importlib.import_module(self.module),self.function)
        return fn(text)

def make_parser(cfg:dict[str,Any],ontology:dict[str,Any])->ParserAdapter:
    backend=require(cfg,"stage1.backend"); version=require(cfg,"stage1.parser_version")
    if backend=="rule": return RuleParser(version,ontology)
    if backend=="openai_compatible":
        return OpenAICompatibleParser(require(cfg,"stage1.openai_compatible.base_url"),require(cfg,"stage1.openai_compatible.model"),require(cfg,"stage1.openai_compatible.api_key_env"),float(require(cfg,"stage1.openai_compatible.timeout_s")),version,ontology)
    if backend=="python_callable": return PythonCallableParser(require(cfg,"stage1.python_callable.module"),require(cfg,"stage1.python_callable.function"))
    raise PipelineFailure("CONFIG_INVALID",f"Unsupported stage1.backend={backend}")
