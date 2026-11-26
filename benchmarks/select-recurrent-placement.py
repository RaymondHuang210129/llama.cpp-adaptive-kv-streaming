#!/usr/bin/env python3
"""Fit and measure fixed recurrent placements before launching a server.

Accepts the arena finder's arguments, plus --placement-reserve-mib and
--previous-selection. Free the GPU first. Each placement is fitted independently;
capacity probes are excluded from timing. One final full-workload measurement is
retained per placement. No server stays running or changes placement in flight.
Use --no-warmup to disable the server's ordinary startup warmup as well.
"""
import argparse
import contextlib
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))


def choose(rows, previous=None, hysteresis=.05):
    if not math.isfinite(hysteresis) or not 0<=hysteresis<1:
        raise ValueError("invalid hysteresis")
    names=set();valid=[]
    for row in rows:
        name=row["provider"]
        if name not in ("native","ram","compact") or name in names:
            raise ValueError("invalid or duplicate provider")
        names.add(name)
        if not row["fits"]:continue
        rate=row["tokens_per_second"]
        if not math.isfinite(rate) or rate<=0:raise ValueError("invalid measured rate")
        valid.append(row)
    if not valid:raise ValueError("no qualified measured placement")
    best=max(valid,key=lambda row:row["tokens_per_second"])
    current=next((row for row in valid if row["provider"]==previous),None)
    if current and best["tokens_per_second"]<=current["tokens_per_second"]/(1-hysteresis):return current
    return best


def load(name):
    spec=importlib.util.spec_from_file_location(name.replace('-','_'),ROOT/(name+'.py'))
    assert spec is not None and spec.loader is not None
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def identity(args, qualification):
    config={key:str(value) if isinstance(value,Path) else [str(p) for p in value] if key=="image" else value
            for key,value in vars(args).items() if key not in ("output","port","initial_arena_mib")}
    files=[args.model,args.model_draft,args.mmproj,args.prefill_text,args.prompt_file,*args.image]
    return {"arguments":config,"files":{str(p.resolve()):qualification.digest(p) for p in files if p},
            "runtime":qualification.runtime_identity(args.server)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placement-reserve-mib",type=int,default=0,
                        help="optional subtraction from the fitted arena; default uses the validated maximum")
    parser.add_argument("--previous-selection",type=Path)
    options,remaining=parser.parse_known_args(argv)
    finder=load("find-max-arena");qualification=load("qualify-recurrent-compact")
    args=finder.arguments(remaining)
    if options.placement_reserve_mib<0 or args.decode_tokens<64:
        parser.error("reserve must be nonnegative and measured decoding needs at least 64 outputs")
    if args.recurrent_rollback_ram or args.recurrent_rollback_compact or args.recurrent_rollback_auto:
        parser.error("omit explicit placement flags; this tool compares all three placements")
    if args.mmproj and not args.image:
        parser.error("supply a concrete image workload; automatic image-capacity search is a separate operation")
    args.output=args.output.resolve();args.server=args.server.resolve(strict=True)
    args.measure_decode=True;args.probe_mode="full";args.validate_selected=False
    inputs=identity(args,qualification)
    inputs["reserve_mib"]=options.placement_reserve_mib
    previous="native"
    if options.previous_selection:
        saved=json.loads(options.previous_selection.read_text())
        if saved["identity"]!=inputs:raise ValueError("previous selection has a different runtime or workload")
        previous=saved["selected"]["provider"]
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=args.output/"identity.json"
    if manifest.exists() and json.loads(manifest.read_text())!=inputs:
        raise ValueError("output directory belongs to different inputs")
    manifest.write_text(json.dumps(inputs,indent=2)+"\n")
    if os.name=="nt":
        print("Windows: configure this server executable to Prefer No Sysmem Fallback in NVIDIA settings before fitting.",flush=True)
    rows=[];reference=None
    row: dict[str, Any]
    record: dict[str, Any]
    for provider in ("native","ram","compact"):
        directory=args.output/provider;directory.mkdir(exist_ok=True)
        result_path=directory/"measured.json"
        variant=copy.deepcopy(args)
        variant.recurrent_rollback_ram=provider=="ram"
        variant.recurrent_rollback_compact=provider=="compact"
        variant.recurrent_rollback_auto=False
        if result_path.exists():
            row=json.loads(result_path.read_text())
        else:
            if list(directory.iterdir()):raise RuntimeError(f"{directory}: inspect incomplete attempt before resuming")
            fitting=copy.deepcopy(variant)
            fitting.output=directory/"fit";fitting.output.mkdir()
            fitting.measure_decode=False;fitting.probe_mode="preflight"
            fitting.validate_selected=options.placement_reserve_mib==0
            fitting.decode_tokens=variant.decode_tokens if fitting.validate_selected else 16
            print("FIT",provider,flush=True)
            try:
                with (directory/"fit.log").open("w") as output,contextlib.redirect_stdout(output):
                    fitted=finder.fit(fitting)
            except RuntimeError as error:
                if not str(error).startswith("no allocatable arena found:"):raise
                row={"provider":provider,"fits":False,"error":str(error)}
                result_path.write_text(json.dumps(row,indent=2)+"\n")
                rows.append(row);continue
            arena=fitted["arena_mib"]-options.placement_reserve_mib
            if arena<=0:raise ValueError("reserve exceeds the fitted arena")
            variant.output=directory/"measurement";variant.output.mkdir()
            from recurrent_measurement import MemorySampler
            print("MEASURE",provider,"arena_mib",arena,flush=True)
            with MemorySampler() as memory:
                try:record=finder.probe(variant,arena)
                except finder.SWEEP.ArenaTooSmallError as error:
                    record={"fits":False,"arena_mib":arena,"error":str(error),"minimum_mib":error.minimum_mib,
                            "command":finder.server_command(variant,arena)}
            record["memory"]=memory.report()
            record_path=variant.output/"record.json"
            record_path.write_text(json.dumps(record,indent=2)+"\n")
            row={"provider":provider,"fits":record["fits"],"record":str(record_path),
                 "maximum_arena_mib":fitted["arena_mib"],"arena_mib":arena,
                 "reserve_mib":options.placement_reserve_mib,"command":record["command"],
                 "compact_observed":record.get("observed_compact_recurrent"),"peak_bytes":record["memory"]["peak_bytes"]}
            if record["fits"]:
                observations=qualification.responses(record,variant.output)
                cohort=observations.get("joint",observations.get("main"))
                if not cohort or not cohort[0]["common_decode_window"]:
                    raise RuntimeError("no measured common decode interval")
                row["tokens_per_second"]=cohort[0]["common_decode_window"]["tokens_per_second"]
            result_path.write_text(json.dumps(row,indent=2)+"\n")
        if row["fits"]:
            record=json.loads(Path(row["record"]).read_text())
            observations=qualification.responses(record,Path(row["record"]).parent)
            if reference is not None:qualification.compare(reference,observations)
            else:reference=observations
        rows.append(row)
        print("RESULT",provider,row.get("tokens_per_second"),flush=True)
    selected=choose(rows,previous)
    report={"identity":inputs,"selected":selected,"measurements":rows,"hysteresis":.05,
            "scope":"One full measurement per independently fitted placement; capacity probes excluded; all servers closed."}
    (args.output/"selection.json").write_text(json.dumps(report,indent=2)+"\n")
    commands=finder.shell_commands(selected["command"])
    (args.output/"command.txt").write_text(commands["powershell" if os.name=="nt" else "posix"]+"\n")
    print("SELECTED",selected["provider"],"arena_mib",selected["arena_mib"],"tokens/s",selected["tokens_per_second"],flush=True)


if __name__=="__main__":main()
