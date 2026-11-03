#!/usr/bin/env python3
"""Host-only tests for vision qualification launch and cleanup contracts."""

import argparse
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import test_adaptive_vision as harness


class Launch(unittest.TestCase):
    def test_finder_admits_qualified_dflash_images_and_preserves_unsupported_width_guard(self):
        root=Path(__file__).resolve().parents[3]
        spec=importlib.util.spec_from_file_location("vision_arena_finder",root/"benchmarks/find-max-arena.py")
        finder=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(finder)
        base=["--model","target.gguf","--model-draft","draft.gguf","--mmproj","projector.gguf",
              "--image","image.png","--probe-mode","preflight"]
        with patch.object(finder.subprocess,"Popen") as process:
            args=finder.arguments(base+["--spec-mode","dflash","--draft-length","3"])
            command=finder.server_command(args,1024)
            self.assertIn("--mmproj",command)
            self.assertIn("--model-draft",command)
            self.assertNotIn("--spec-draft-type-k",command)
            self.assertEqual(args.image,[Path("image.png")])
            mixed=finder.arguments(base+["--spec-mode","dflash-k4v","--ngram-m","7"])
            self.assertEqual(mixed.spec_mode,"dflash-k4v")
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                finder.arguments(base+["--spec-mode","dflash-k4v","--ngram-m","8"])
            process.assert_not_called()

    def test_image_state_actions_require_complete_files_and_matching_counts(self):
        directory=Path(self.directory.name); path=directory/"vision.bin"; path.write_bytes(b"exact-state")
        saved={"n_saved":17,"n_written":path.stat().st_size}
        with patch.object(harness,"request",return_value=(200,saved)):
            self.assertEqual(harness.vision_state_action(1,directory,"save"),saved)
        with patch.object(harness,"request",return_value=(200,{"n_restored":17,"n_read":saved["n_written"]})):
            harness.vision_state_action(1,directory,"restore",saved)
        for response in ({"n_restored":16,"n_read":saved["n_written"]},{"n_restored":17,"n_read":1}):
            with patch.object(harness,"request",return_value=(200,response)),self.assertRaises(RuntimeError):
                harness.vision_state_action(1,directory,"restore",saved)
        path.write_bytes(b"short")
        with patch.object(harness,"request",return_value=(200,saved)),self.assertRaises(RuntimeError):
            harness.vision_state_action(1,directory,"save")

    def test_image_completion_requires_real_drafts_and_exact_output_accounting(self):
        result={"tokens":[2,3],"timings":{"predicted_n":2,"draft_n":3,"draft_n_accepted":1,"cache_n":0,"prompt_n":64}}
        self.assertEqual(harness.vision_observation(result,2,True)["tokens"],[2,3])
        import copy
        for field,value in (("predicted_n",1),("draft_n",0),("draft_n_accepted",4),("cache_n",-1),("prompt_n",0)):
            bad=copy.deepcopy(result);bad["timings"][field]=value
            with self.subTest(field=field),self.assertRaises(RuntimeError):
                harness.vision_observation(bad,2,True)

    def test_lifecycle_restart_keeps_the_same_slot_directory(self):
        self.args.lifecycle=True
        for mode in ("arena-state","arena-restart"):
            with self.launch("vision arena rejects this configuration"),patch.object(harness.os,"killpg",create=True):
                with harness.server(self.args,mode,reject=True):pass
        directories=[]
        for command,_ in self.commands:
            self.assertIn("--slot-save-path",command)
            directories.append(command[command.index("--slot-save-path")+1])
        self.assertEqual(directories[0],directories[1])

    def test_dflash_startup_rejections_accept_the_specific_common_diagnostics(self):
        messages=("adaptive DFlash2 with mmproj does not support K4V image history",
                  "adaptive DFlash2 supports 1-7 draft tokens")
        for message in messages:
            with self.subTest(message=message),self.launch(message),patch.object(harness.os,"killpg",create=True):
                with harness.server(self.args,"arena-reject-dflash",reject=True):pass

    def test_a_vision_banner_does_not_turn_device_faults_into_config_rejections(self):
        for error in ("cudaMalloc failed: out of memory","illegal memory access","device-side assert","no kernel image"):
            with self.subTest(error=error),self.launch("vision arena enabled\n"+error),patch.object(harness.os,"killpg",create=True):
                with self.assertRaises(AssertionError):
                    with harness.server(self.args,"arena-reject-fault",reject=True):pass

    def test_startup_negative_profiles_do_not_reject_supported_cache_formats(self):
        cases=harness.vision_startup_rejections()
        self.assertEqual(len(cases),6)
        self.assertIn(('kv-offload',('--no-kv-offload',)),cases)
        for _,flags in cases:
            self.assertNotIn('-ctk',flags)
            self.assertNotIn('--cache-type-k',flags)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.args = argparse.Namespace(server=root/'server', model=root/'model', mmproj=root/'projector',
            output=root, context=8192, batch_size=64, ubatch_size=64, cache_ram_mib=2048,
            arena_mib=1024, mtp_length=3, stock_uvm=False)
        self.commands = []

    def launch(self, diagnostic, timeout=False):
        def popen(command, **kwargs):
            self.commands.append((command, kwargs))
            kwargs['stdout'].write(diagnostic)
            kwargs['stdout'].flush()
            class Process:
                pid = 987654
                waits = 0
                def poll(self):
                    return None
                def wait(self, timeout=None):
                    self.waits += 1
                    if self.waits == 1 and timeout:
                        raise subprocess.TimeoutExpired(command, timeout)
                    return 1
                def terminate(self):
                    pass
                def kill(self):
                    pass
            process = Process()
            if not timeout:
                process.waits = 1
            return process
        return patch.object(harness.subprocess, 'Popen', side_effect=popen)

    def test_mixed_dflash_vision_keeps_independent_limits_and_inherited_cache(self):
        self.args.mtp_length=0;self.args.dflash_length=3;self.args.model_draft=Path("draft")
        self.args.ngram_m=7;self.args.ngram_n=2;self.args.ngram_min_hits=1
        for mode in ("stock","arena"):
            flags=harness.speculative_arguments(self.args,mode)
            self.assertEqual(flags[flags.index("--spec-type")+1],"ngram-map-k4v,draft-dflash")
            self.assertEqual(flags[flags.index("--spec-draft-n-max")+1],"3")
            self.assertEqual(flags[flags.index("--spec-ngram-map-k4v-size-m")+1],"7")
            self.assertEqual("--spec-draft-type-k" in flags,mode=="stock")

    def test_mixed_vision_policy_errors_reject_before_launch(self):
        self.args.mtp_length=0;self.args.dflash_length=3;self.args.model_draft=Path("draft")
        for n,k,hits in [(0,7,1),(1025,7,1),(2,8,1),(2,-1,1),(2,7,0)]:
            self.args.ngram_n=n;self.args.ngram_m=k;self.args.ngram_min_hits=hits
            with self.subTest(policy=(n,k,hits)),patch.object(harness.subprocess,"Popen") as process:
                with self.assertRaises(ValueError):harness.speculative_arguments(self.args,"arena")
                process.assert_not_called()

    def test_image_source_accounting_rejects_a_replayed_correction_as_an_accept(self):
        value={"tokens":[17,17],"timings":{"predicted_n":2,"prompt_n":64,"cache_n":0,"draft_n":7,"draft_n_accepted":0}}
        stats={"ngram-map-k4v":{"generated_tokens":7,"accepted_tokens":0},"draft-dflash":{"generated_tokens":0,"accepted_tokens":0}}
        self.assertEqual(harness.vision_observation(value,2,True,stats)["source_stats"],stats)
        import copy
        for field in ("generated_tokens","accepted_tokens"):
            bad=copy.deepcopy(stats);bad["ngram-map-k4v"][field]+=1
            with self.subTest(field=field),self.assertRaises(RuntimeError):harness.vision_observation(value,2,True,bad)

    def test_native_dflash_uses_matching_sidecar_and_cache_types(self):
        self.args.mtp_length = 0
        self.args.dflash_length = 3
        self.args.model_draft = Path(self.directory.name)/'draft.gguf'
        with self.launch('vision arena rejects this configuration'), patch.object(harness.os,'killpg',create=True):
            with harness.server(self.args,'stock',reject=True):
                pass
        command,_ = self.commands[0]
        self.assertIn('--model-draft',command)
        self.assertEqual(command[command.index('--model-draft')+1],str(self.args.model_draft.resolve()))
        modes=[command[i+1] for i,value in enumerate(command) if value=='--spec-type']
        self.assertEqual(modes,['draft-dflash'])
        self.assertEqual(command[command.index('--spec-draft-n-max')+1],'3')
        self.assertEqual(command[command.index('--spec-draft-type-k')+1],'q8_0')
        self.assertEqual(command[command.index('--spec-draft-type-v')+1],'q4_0')

    def test_arena_dflash_inherits_target_cache_types(self):
        self.args.mtp_length = 0
        self.args.dflash_length = 7
        self.args.model_draft = Path(self.directory.name)/'draft.gguf'
        with self.launch('vision arena rejects this configuration'), patch.object(harness.os,'killpg',create=True):
            with harness.server(self.args,'arena-reject-dflash',reject=True):
                pass
        command,_ = self.commands[0]
        self.assertIn('--model-draft',command)
        self.assertNotIn('--spec-draft-type-k',command)
        self.assertNotIn('--spec-draft-type-v',command)

    def test_invalid_dflash_configuration_is_rejected_before_process_launch(self):
        for mtp,draft,length in ((0,None,3),(3,Path('draft.gguf'),3),(0,Path('draft.gguf'),8)):
            with self.subTest(mtp=mtp,draft=draft,length=length), patch.object(harness.subprocess,'Popen') as launch:
                self.args.mtp_length=mtp
                self.args.model_draft=draft
                self.args.dflash_length=length
                with self.assertRaises(ValueError):
                    with harness.server(self.args,'stock'):
                        pass
                launch.assert_not_called()

    def test_launch_sets_matching_mtp_kv_and_arena_disables_inherited_uvm(self):
        with self.launch('vision arena rejects this configuration'), patch.dict(os.environ, {'GGML_CUDA_ENABLE_UNIFIED_MEMORY': '1'}), \
                patch.object(harness.os, 'killpg', create=True):
            with harness.server(self.args, 'arena-reject-parallel', reject=True):
                pass
        command, kwargs = self.commands[0]
        self.assertEqual(command[command.index('--spec-draft-n-max')+1], '3')
        self.assertNotIn('--spec-draft-type-k',command)
        self.assertNotIn('--spec-draft-type-v',command)
        self.assertNotIn('GGML_CUDA_ENABLE_UNIFIED_MEMORY', kwargs['env'])

    def test_native_mtp_control_still_selects_matching_independent_cache_types(self):
        with self.launch('vision arena rejects this configuration'), patch.object(harness.os,'killpg',create=True):
            with harness.server(self.args,'stock',reject=True):
                pass
        command,_=self.commands[0]
        self.assertEqual(command[command.index('--spec-draft-type-k')+1],'q8_0')
        self.assertEqual(command[command.index('--spec-draft-type-v')+1],'q4_0')

    def test_stock_uvm_is_explicit_and_does_not_enable_arena_uvm(self):
        self.args.stock_uvm = True
        for mode in ('stock', 'arena-reject-parallel'):
            with self.launch('vision arena rejects this configuration'), patch.object(harness.os, 'killpg', create=True):
                with harness.server(self.args, mode, reject=True):
                    pass
        self.assertEqual(self.commands[0][1]['env']['GGML_CUDA_ENABLE_UNIFIED_MEMORY'], '1')
        self.assertNotIn('GGML_CUDA_ENABLE_UNIFIED_MEMORY', self.commands[1][1]['env'])

    def test_arena_uvm_is_explicit_and_does_not_enable_stock_uvm(self):
        self.args.arena_uvm = True
        for mode in ('arena-reject-parallel', 'stock'):
            with self.launch('vision arena rejects this configuration'), patch.object(harness.os, 'killpg', create=True):
                with harness.server(self.args, mode, reject=True):
                    pass
        self.assertEqual(self.commands[0][1]['env']['GGML_CUDA_ENABLE_UNIFIED_MEMORY'], '1')
        self.assertNotIn('GGML_CUDA_ENABLE_UNIFIED_MEMORY', self.commands[1][1]['env'])

    def test_mtp_cli_rejection_is_accepted_before_vision_admission(self):
        with self.launch('error: attached MTP KV streaming supports at most 3 draft tokens'), \
                patch.object(harness.os, 'killpg', create=True):
            with harness.server(self.args, 'arena-reject-mtp-length', reject=True):
                pass

    def test_unrelated_vision_oom_is_not_a_configuration_rejection(self):
        with self.launch('CUDA out of memory while loading vision weights'), patch.object(harness.os, 'killpg', create=True):
            with self.assertRaises(AssertionError):
                with harness.server(self.args, 'arena-reject-fit', reject=True):
                    pass

    @unittest.skipUnless(os.name == 'posix', 'process-group cleanup is POSIX-only')
    def test_cleanup_stops_the_whole_instrumented_server_group(self):
        with self.launch('', timeout=True), patch.object(harness.os, 'killpg') as kill, \
                patch.object(harness, 'request', return_value=(200, {})):
            with self.assertRaisesRegex(RuntimeError, 'caller failure'):
                with harness.server(self.args, 'arena'):
                    raise RuntimeError('caller failure')
        self.assertTrue(self.commands[0][1]['start_new_session'])
        self.assertIn((987654, signal.SIGTERM), [call.args for call in kill.call_args_list])
        self.assertIn((987654, signal.SIGKILL), [call.args for call in kill.call_args_list])


class FinderCapacity(unittest.TestCase):
    def finder(self):
        root=Path(__file__).resolve().parents[3]
        spec=importlib.util.spec_from_file_location("capacity_finder",root/"benchmarks/find-max-arena.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def mixed_args(self,module,directory):
        args=module.arguments(["--model","model","--model-draft","draft","--spec-mode","dflash-k4v",
            "--draft-length","3","--ngram-n","2","--ngram-m","7","--ngram-min-hits","1",
            "--context-size","4096","--decode-tokens","8","--probe-mode","preflight"])
        args.output=Path(directory);args.mmproj=Path("projector")
        return args

    def test_mixed_vision_cli_uses_the_same_inherited_cache_and_independent_widths(self):
        module=self.finder()
        for images in ([],["--image","photo.png"]):
            args=module.arguments(["--model","model","--model-draft","draft","--mmproj","projector",
                "--spec-mode","dflash-k4v","--draft-length","3","--ngram-m","7",*images])
            command=module.server_command(args,1024)
            self.assertIn("--mmproj",command);self.assertNotIn("--spec-draft-type-k",command)
            self.assertEqual(command[command.index("--spec-ngram-map-k4v-size-m")+1],"7")

    def test_image_route_proof_retains_images_and_never_slices_their_prompt(self):
        module=self.finder()
        with tempfile.TemporaryDirectory() as directory:
            args=self.mixed_args(module,directory);calls=[]
            def response(url,payload,**kwargs):
                if url.endswith("/tokenize"):
                    return {"tokens":[17]*(len(payload["content"])//2)}
                calls.append(payload);count=payload["n_predict"]
                return {"tokens":[17]*count,"timings":{"predicted_n":count,"prompt_n":96,"draft_n":3,"draft_n_accepted":2,"cache_n":0}}
            with patch.object(module,"request_json",side_effect=response):
                proof=module.mixed_image_route_probe(args,"http://fixture",["same-image"],1024)
            self.assertEqual(len(calls),2)
            self.assertEqual([r["route"] for r in proof["requests"]],["image-miss","image-copy"])
            for call in calls:
                self.assertEqual(call["prompt"]["multimodal_data"],["same-image"])
                self.assertEqual(call["prompt"]["prompt_string"].count(module.MARKER),1)
                self.assertEqual(call["logit_bias"],[[17,1000]])
            self.assertTrue(calls[0]["prompt"]["prompt_string"].startswith(" B"))
            self.assertTrue(calls[1]["prompt"]["prompt_string"].endswith(" B"))

    def test_image_route_proof_rejects_bad_tokenization_and_insufficient_context_before_completion(self):
        module=self.finder()
        with tempfile.TemporaryDirectory() as directory:
            args=self.mixed_args(module,directory)
            with patch.object(module,"request_json",return_value={"tokens":[1,2]}) as request:
                with self.assertRaisesRegex(ValueError,"single ordinary token"):
                    module.mixed_image_route_probe(args,"http://fixture",["image"],1024)
                self.assertEqual(request.call_count,1)
            args.ngram_n=1024
            with patch.object(module,"request_json",return_value={"tokens":[17]}) as request:
                with self.assertRaisesRegex(ValueError,"proof history|context"):
                    args.context_size=512
                    module.mixed_image_route_probe(args,"http://fixture",["image"],1024)
                self.assertLessEqual(request.call_count,1)

    @staticmethod
    def mixed_statistics(ordinal,copied,neural,accepted_copy,accepted_neural,forwards):
        lines=[]
        for name,generated,accepted,pred in [("ngram-map-k4v",copied,accepted_copy,0),("draft-dflash",neural,accepted_neural,forwards)]:
            lines.append(f"statistics {name}: #calls(b,g,a) = {ordinal} {ordinal} {ordinal}, #gen drafts = {ordinal}, #acc drafts = 0, #gen tokens = {generated}, #acc tokens = {accepted}, #prediction forwards = {pred}")
        return "\n".join(lines)

    def test_mixed_image_accounting_cannot_borrow_copy_or_forward_proof_from_text_prelude(self):
        module=self.finder();completed=[{"label":"text-prelude","timings":{"draft_n":3,"draft_n_accepted":2}},
            {"label":"image-copy","timings":{"draft_n":7,"draft_n_accepted":7}},
            {"label":"image-miss","timings":{"draft_n":3,"draft_n_accepted":0}}]
        log="\n".join([self.mixed_statistics(1,0,3,0,2,1),self.mixed_statistics(2,7,3,7,2,1),self.mixed_statistics(3,7,6,7,2,2)])
        result=module.validate_completed_image_sources(log,completed)
        self.assertEqual(result["ngram-map-k4v"]["generated_tokens"],7)
        self.assertEqual(completed[1]["source_stats"]["ngram-map-k4v"]["generated_tokens"],7)
        for bad in (log.replace("#gen tokens = 7","#gen tokens = 0"),
                log.replace("#prediction forwards = 2","#prediction forwards = 1"),log.splitlines()[2:],log.replace("#gen tokens = 6","#gen tokens = 7")):
            if isinstance(bad,list):bad="\n".join(bad)
            with self.subTest(log=bad),self.assertRaises((RuntimeError,ValueError)):
                module.validate_completed_image_sources(bad,completed)

    def test_mixed_image_runtime_probe_accounts_text_prelude_and_all_encoder_phases(self):
        from types import SimpleNamespace
        module,plan=self.plan();shape=plan["ratios"][0]["candidates"][0]
        with tempfile.TemporaryDirectory() as directory:
            args=self.mixed_args(module,directory);args.port=0
            args.image=[args.output/"image.png"];args.image[0].write_bytes(b"image")
            args.prefill_text=args.output/"corpus";args.prefill_text.write_text("corpus")
            args._expected_vision_shape=shape;args._vision_validate_text_first=True
            phase="vision_phase: parent=1073741824 image_width=256 image_height=256 image_tokens=64 weights=100 compute=200 graph=150 operator=25 library=20 padding=5 borrowed_after=0 suspended_after=0 diagnostics=1"
            stats="\n".join([self.mixed_statistics(1,0,3,0,2,1),self.mixed_statistics(2,0,6,0,4,2),
                self.mixed_statistics(3,0,9,0,6,3),self.mixed_statistics(4,7,9,7,6,3)])
            def start(command,**kwargs):
                kwargs["stdout"].write("\n".join([phase]*3)+"\n"+stats);kwargs["stdout"].flush()
                return SimpleNamespace(pid=123)
            calls=[]
            def response(url,body,**kwargs):
                if url.endswith("/tokenize"):
                    return {"tokens":list(range(2048)) if body["content"]=="corpus" else [17]*(len(body["content"])//2)}
                calls.append(body);count=body["n_predict"];copy=len(calls)==4
                return {"tokens":[17]*count,"timings":{"prompt_n":2048 if len(calls)==1 else 96,"predicted_n":count,
                    "cache_n":0,"draft_n":7 if copy else 3,"draft_n_accepted":7 if copy else 2}}
            with patch.object(module.subprocess,"Popen",side_effect=start),patch.object(module.SWEEP,"wait_ready"),patch.object(module.SWEEP,"stop_process"),patch.object(module.SWEEP,"validate_execution_log"),patch.object(module,"request_json",side_effect=response):
                result=module._runtime_probe(args,1024)
            self.assertTrue(result["fits"],result["error"])
            self.assertEqual(len(calls),4)
            self.assertEqual([row["label"] for row in result["completed_requests"]],["text-prelude","image","image-miss","image-copy"])
            self.assertEqual(result["mixed_proof"]["source_stats"]["draft-dflash"]["generated_tokens"],9)

    def plan(self):
        module=self.finder()
        plan={"schema":"vision-memory-v2","controlled_workspace":["graph","operator","library"],
              "opaque_runtime_allocations_included":False,"weight_bytes":100,
              "arena_bytes":1024*1048576,"family":"qwen3vl","alignment":32,
              "min_image_tokens":64,"effective_image_max_tokens":256,"images_per_probe":1,"microbatch":256,"ratios":[]}
        for ratio in ("1:1","4:3","3:4","16:9","9:16","2:1","1:2"):
            p,q=map(int,ratio.split(":"))
            plan["ratios"].append({"ratio":ratio,"shapes_measured":1,"candidates":[{"width":256,"height":256,
                "input_width":p*256,"input_height":q*256,"image_tokens":64,"image_max_tokens":64,
                "weight_bytes":100,"compute_bytes":200,"workspace":{"graph_bytes":150,"operator_bytes":25,
                "library_bytes":20,"padding_bytes":5,"total_bytes":200},"required_arena_bytes":400,"shape_id":1}],"budget_shape_ids":[1]*193})
        return module,plan

    def test_controlled_workspace_report_accepted(self):
        module,plan=self.plan()
        self.assertIs(module.validate_vision_plan(plan,1024),plan)

    def test_graph_only_reports_cannot_admit_complete_vision(self):
        module,plan=self.plan();plan["schema"]="vision-memory-v1"
        plan.pop("controlled_workspace");plan.pop("opaque_runtime_allocations_included")
        for row in plan["ratios"]:
            for shape in row["candidates"]:shape.pop("workspace")
        with self.assertRaises(RuntimeError):module.validate_vision_plan(plan,1024)

    def test_vision_execution_requires_all_planned_components(self):
        module,plan=self.plan();shape=plan["ratios"][0]["candidates"][0]
        legacy="vision_phase: parent=1073741824 image_width=256 image_height=256 image_tokens=64 weights=100 compute=200 borrowed_after=0 suspended_after=0 diagnostics=1"
        with self.assertRaises(RuntimeError):module.check_vision_phase(legacy,shape,1024)
        complete=legacy+" graph=150 operator=25 library=20 padding=5"
        for bad in (complete.replace("library=20","library=0"),complete.replace("operator=25","operator=24"),complete.replace("padding=5","padding=4")):
            with self.assertRaises(RuntimeError):module.check_vision_phase(bad,shape,1024)

    def test_workspace_components_reject_missing_bad_or_inconsistent_totals(self):
        import copy
        module,original=self.plan()
        for mutation in range(8):
            plan=copy.deepcopy(original);shape=plan["ratios"][0]["candidates"][0]
            if mutation==0:shape.pop("workspace")
            elif mutation==1:shape["workspace"]["total_bytes"]+=1
            elif mutation==2:shape["workspace"]["operator_bytes"]=-1
            elif mutation==3:shape["workspace"]["library_bytes"]=True
            elif mutation==4:shape["workspace"].pop("padding_bytes")
            elif mutation==5:plan["controlled_workspace"]=["graph"]
            elif mutation==6:plan["opaque_runtime_allocations_included"]=True
            else:shape["weight_bytes"]+=1
            with self.subTest(mutation=mutation),self.assertRaises(RuntimeError):module.validate_vision_plan(plan,1024)

    def test_capacity_output_contains_landscape_portrait_and_actual_pixels(self):
        module,plan=self.plan()
        module.validate_vision_plan(plan,1024)
        report={"ratios":[{"ratio":row["ratio"],"maximum":row["candidates"][0]} for row in plan["ratios"]]}
        output=module.format_vision_capacity(report)
        for ratio in ("1:1","4:3","3:4","16:9","9:16","2:1","1:2"):self.assertIn(ratio,output)
        self.assertIn("256x256",output)
        self.assertIn("--image-max-tokens 64",output)

    def test_capacity_rejects_incomplete_or_impossible_server_plans(self):
        import copy
        module,valid=self.plan()
        for change in range(7):
            plan=copy.deepcopy(valid)
            if change==0:plan["ratios"].pop()
            elif change==1:plan["arena_bytes"]-=1
            elif change==2:plan["ratios"][0]["candidates"][0]["width"]=255
            elif change==3:plan["ratios"][0]["candidates"][0]["required_arena_bytes"]=1025*1048576
            elif change==4:plan["ratios"][0]["candidates"][0]["image_tokens"]=65
            elif change==5:plan["ratios"][0]["candidates"][0]["height"]=True
            else:plan["schema"]="unknown"
            with self.subTest(change=change),self.assertRaises(RuntimeError):module.validate_vision_plan(plan,1024)

    def test_vision_boundary_cannot_pass_with_silently_smaller_encoded_shape(self):
        module,plan=self.plan();shape=plan["ratios"][0]["candidates"][0]
        line="vision_phase: parent=1073741824 image_width=256 image_height=256 image_tokens=64 weights=100 compute=200 graph=150 operator=25 library=20 padding=5 borrowed_after=0 suspended_after=0 diagnostics=1"
        self.assertEqual(module.check_vision_phase(line,shape,1024)["image_tokens"],64)
        for bad in (line.replace("image_width=256","image_width=128"),line.replace("compute=200","compute=201"),line+"\n"+line):
            with self.assertRaises(RuntimeError):module.check_vision_phase(bad,shape,1024)

    def test_cli_requests_automatic_vision_capacity_without_an_image(self):
        module=self.finder()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ("server","model","projector","corpus"):(root/name).write_text("fixture")
            report={"shell":"posix","shell_command":"server"}
            options=["--server",str(root/"server"),"--model",str(root/"model"),"--mmproj",str(root/"projector"),
                     "--prefill-text",str(root/"corpus"),"--output",str(root/"result")]
            with patch.object(module,"fit",return_value=report) as fit,contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(options),0)
            self.assertTrue(fit.call_args.kwargs["report_vision"])
            self.assertEqual(fit.call_args.args[0].image,[])

    def test_auto_probe_keeps_the_text_workload_and_adds_a_minimum_image(self):
        module,plan=self.plan()
        args=module.arguments(["--model","model","--mmproj","projector"])
        with tempfile.TemporaryDirectory() as directory:
            args.output=Path(directory)
            with patch.object(module,"vision_plan_probe",return_value=plan),patch.object(module,"_runtime_probe",return_value={"fits":True}) as probe:
                result=module.probe_with_vision(args,1024)
            config=probe.call_args.args[0]
            self.assertEqual(config.image,[])
            self.assertEqual(config.image_max_tokens,64)
            self.assertEqual(len(config._vision_bootstrap),7)
            self.assertEqual({row["ratio"] for row in config._vision_bootstrap},set(module.VISION_RATIOS))
            self.assertTrue(all(Path(row["image"]).is_file() for row in config._vision_bootstrap))
            self.assertIsNone(args.image_max_tokens)
            self.assertEqual(result["vision_plan"],plan)

    def test_ratio_search_does_not_downgrade_numerical_failures_into_capacity(self):
        module,plan=self.plan()
        args=module.arguments(["--model","model","--mmproj","projector"])
        with tempfile.TemporaryDirectory() as directory:
            args.output=Path(directory)
            with patch.object(module,"probe",side_effect=RuntimeError("numerical mismatch")) as probe:
                with self.assertRaisesRegex(RuntimeError,"numerical mismatch"):module.vision_capacity(args,1024,plan)
            self.assertEqual(probe.call_count,1)

    def test_explicit_images_keep_existing_default_preprocessing_limits(self):
        module=self.finder()
        args=module.arguments(["--model","model","--mmproj","projector","--image","photo.png"])
        self.assertEqual((args.image_min_tokens,args.image_max_tokens),(64,4096))
        auto=module.arguments(["--model","model","--mmproj","projector"])
        self.assertIsNone(auto.image_min_tokens);self.assertIsNone(auto.image_max_tokens)

    def test_common_launch_budget_is_admitted_and_validated_for_every_ratio(self):
        module,plan=self.plan();plan["effective_image_max_tokens"]=66
        for index,row in enumerate(plan["ratios"]):
            row["budget_shape_ids"]=[1,1,-1] if index==0 else [1,1,1]
            row["candidates"][0]["image_max_tokens"]=65 if index==0 else 66
        args=module.arguments(["--model","model","--mmproj","projector"])
        observed=[]
        def probe(config,arena):
            observed.append(config.image_max_tokens)
            return {"fits":True,"arena_mib":arena}
        with tempfile.TemporaryDirectory() as directory:
            args.output=Path(directory)
            with patch.object(module,"probe",side_effect=probe),patch.object(module,"vision_common_sequence",return_value={"fits":True}):result=module.vision_capacity(args,1024,plan)
        self.assertEqual(result["common_image_max_tokens"],65)
        self.assertTrue(result["common"]["validated"])
        self.assertEqual(len(result["common"]["ratios"]),7)
        self.assertTrue(all(row["maximum"]["image_max_tokens"]==65 for row in result["common"]["ratios"]))
        self.assertIn(66,observed)
        self.assertIn(65,observed)

    def test_runtime_boundary_search_checks_fit_and_rejected_neighbor_with_bounded_restarts(self):
        module=self.finder();observed=[]
        def check(value):
            observed.append(value)
            return value<=37
        self.assertEqual(module.vision_runtime_boundary(list(range(1000)),check),37)
        self.assertIn(38,observed)
        self.assertIn(999,observed)
        self.assertLessEqual(len(observed),12)
        self.assertIsNone(module.vision_runtime_boundary([1,2],lambda _:False))
        self.assertEqual(module.vision_runtime_boundary([1,2],lambda _:True),2)

    def test_vision_shared_mtp_rejection_reports_its_quota_lower_bound(self):
        module=self.finder();args=module.arguments(["--model","model","--mmproj","projector"]);args.port=0
        text=("MTP graph exceeds shared arena quota: available=1073741824 required=1254383744 additional=180641920 bytes; trying a separate workspace\n"
              "unsupported vision arena model/projector capabilities\n")
        with tempfile.TemporaryDirectory() as directory:
            args.output=Path(directory)
            class Process:
                def wait(self,**kwargs):return 1
            def launch(command,**kwargs):
                kwargs["stderr"].write(text);kwargs["stderr"].flush();return Process()
            with patch.object(module.subprocess,"Popen",side_effect=launch),patch.object(module.SWEEP,"stop_process"),self.assertRaises(module.SWEEP.ArenaTooSmallError) as failure:
                module.vision_plan_probe(args,1024)
            self.assertEqual(failure.exception.minimum_mib,(1254383744+1048575)//1048576)

    def test_failed_planner_keeps_upper_boundary_artifacts(self):
        module=self.finder();args=module.arguments(["--model","model","--mmproj","projector"])
        with tempfile.TemporaryDirectory() as directory:
            args.output=Path(directory)
            log=args.output/"arena-1000-vision-plan.log";log.write_text("CUDA out of memory")
            (args.output/"arena-1000-vision-plan.json").write_bytes(b"")
            with patch.object(module,"vision_plan_probe",return_value=None):result=module.probe_with_vision(args,1000)
            self.assertEqual(result["log"],str(log))
            self.assertEqual(result["log_sha256"],module.hashlib.sha256(log.read_bytes()).hexdigest())
            self.assertIn("--vision-memory-report",result["command"])

    def test_no_common_budget_cannot_publish_a_generic_vision_command(self):
        module,plan=self.plan();args=module.arguments(["--model","model","--mmproj","projector"])
        def search(start,check):
            self.assertTrue(check(1024));return 1024
        with tempfile.TemporaryDirectory() as directory:
            args.output=Path(directory)
            with patch.object(module.SWEEP,"find_max_arena_mib",side_effect=search),patch.object(module,"probe_with_vision",return_value={"fits":True,"arena_mib":1024,"timings":{"predicted_n":256},"vision_plan":plan}),patch.object(module,"vision_capacity",return_value={"common_image_max_tokens":None,"common":None,"all_ratios_validated":True}):
                with self.assertRaisesRegex(RuntimeError,"common"):module.fit(args,report_vision=True)
            self.assertFalse((args.output/"command.txt").exists())

    def full_vision_fit(self, succeeds=True):
        module,plan=self.plan();args=module.arguments(["--model","model","--mmproj","projector","--probe-mode","preflight","--validate-selected"])
        def search(start,check):
            self.assertTrue(check(1024));self.assertFalse(check(1025));return 1024
        def bootstrap(config,arena):
            config=argparse.Namespace(**vars(config));config.image_min_tokens=64;config.image_max_tokens=64
            return {"fits":arena==1024,"arena_mib":arena,"timings":{"predicted_n":256},"vision_plan":plan,"command":module.server_command(config,arena)}
        def runtime(config,arena):
            self.assertEqual(config.probe_mode,"full")
            self.assertEqual(len(config._vision_bootstrap),7)
            self.assertEqual(module.probe_text_tokens(config),module.requested_text_tokens(config))
            return {"fits":succeeds,"arena_mib":arena,"probe_mode":config.probe_mode,"timings":{"predicted_n":256},"command":module.server_command(config,arena)}
        return module,args,search,bootstrap,runtime, {"common_image_max_tokens":128,"all_ratios_validated":True}

    def test_selected_full_vision_validates_the_published_image_setting(self):
        module,args,search,bootstrap,runtime,capacity=self.full_vision_fit()
        with tempfile.TemporaryDirectory() as directory:
            args.output=Path(directory)
            with patch.object(module.SWEEP,"find_max_arena_mib",side_effect=search),patch.object(module,"probe_with_vision",side_effect=bootstrap),patch.object(module,"vision_capacity",return_value=capacity),patch.object(module,"_runtime_probe",side_effect=runtime):
                result=module.fit(args,report_vision=True)
            self.assertTrue(result["full_workload_validated"])
            self.assertEqual(result["selected_validation"]["command"],result["command"])

    def test_full_search_also_checks_the_final_common_image_setting(self):
        module,args,search,bootstrap,runtime,capacity=self.full_vision_fit()
        args.probe_mode="full";args.validate_selected=False
        with tempfile.TemporaryDirectory() as directory:
            args.output=Path(directory)
            with patch.object(module.SWEEP,"find_max_arena_mib",side_effect=search),patch.object(module,"probe_with_vision",side_effect=bootstrap),patch.object(module,"vision_capacity",return_value=capacity),patch.object(module,"_runtime_probe",side_effect=runtime):
                result=module.fit(args,report_vision=True)
            self.assertEqual(result["selected_validation"]["command"],result["command"])

    def test_failed_published_full_vision_keeps_receipt_without_command(self):
        module,args,search,bootstrap,runtime,capacity=self.full_vision_fit(False)
        with tempfile.TemporaryDirectory() as directory:
            args.output=Path(directory)
            with patch.object(module.SWEEP,"find_max_arena_mib",side_effect=search),patch.object(module,"probe_with_vision",side_effect=bootstrap),patch.object(module,"vision_capacity",return_value=capacity),patch.object(module,"_runtime_probe",side_effect=runtime):
                with self.assertRaisesRegex(RuntimeError,"full workload validation"):module.fit(args,report_vision=True)
            self.assertFalse((args.output/"command.txt").exists())
            self.assertFalse(module.json.loads((args.output/"selected-validation/record.json").read_text())["fits"])

    def test_full_common_sequence_preserves_requested_history_and_budget(self):
        module,plan=self.plan();args=module.arguments(["--model","model","--mmproj","projector","--context-size","8192"])
        with tempfile.TemporaryDirectory() as directory:
            args.output=Path(directory)
            with patch.object(module,"_runtime_probe",return_value={"fits":True}) as probe:
                module.vision_common_sequence(args,1024,plan,128,args.output/"sequence",probe_mode="full")
            config=probe.call_args.args[0]
            self.assertEqual(module.probe_text_tokens(config),8192-args.decode_tokens-15)
            self.assertEqual(config.image_max_tokens,128)
            self.assertEqual(len(config._vision_bootstrap),7)

    def test_supplied_image_cli_keeps_the_complete_request_mode(self):
        module=self.finder()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ("server","model","projector","photo.png","corpus"):(root/name).write_bytes(b"fixture")
            options=["--server",str(root/"server"),"--model",str(root/"model"),"--mmproj",str(root/"projector"),
                     "--image",str(root/"photo.png"),"--prefill-text",str(root/"corpus"),"--output",str(root/"out")]
            with patch.object(module,"fit",return_value={"shell":"posix","shell_command":"server"}) as fit,contextlib.redirect_stdout(io.StringIO()):module.main(options)
            self.assertFalse(fit.call_args.kwargs["report_vision"])

    def test_common_budget_also_executes_all_ratios_serially_in_one_server(self):
        module,plan=self.plan();args=module.arguments(["--model","model","--mmproj","projector"])
        with tempfile.TemporaryDirectory() as directory:
            args.output=Path(directory)
            with patch.object(module,"_runtime_probe",return_value={"fits":True}) as probe:
                receipt=module.vision_common_sequence(args,1024,plan,64,args.output/"sequence")
            config=probe.call_args.args[0]
            self.assertEqual(config.image,[])
            self.assertEqual(config.probe_mode,"preflight")
            self.assertEqual(len(config._vision_bootstrap),7)
            self.assertTrue(all(row["shape"]["image_max_tokens"]==64 for row in config._vision_bootstrap))
            self.assertTrue(Path(receipt["record"]).is_file())
            with patch.object(module,"_runtime_probe",side_effect=RuntimeError("numerical mismatch")):
                with self.assertRaisesRegex(RuntimeError,"numerical mismatch"):
                    module.vision_common_sequence(args,1024,plan,64,args.output/"failed")

    def test_full_common_sequence_preserves_optional_text_resource_probe(self):
        module,plan=self.plan();args=module.arguments(["--model","model","--mmproj","projector","--copy-probe"])
        with tempfile.TemporaryDirectory() as directory:
            args.output=Path(directory)
            for mode in ("preflight","full"):
                with self.subTest(mode=mode),patch.object(module,"_runtime_probe",return_value={"fits":True}) as probe:
                    module.vision_common_sequence(args,1024,plan,128,args.output/mode,probe_mode=mode)
                    config=probe.call_args.args[0]
                    self.assertEqual(config.copy_probe,mode=="full")
                    if mode=="full":
                        expected=argparse.Namespace(**vars(args));expected.image_min_tokens=64;expected.image_max_tokens=128
                        self.assertEqual(module.server_command(config,1024),module.server_command(expected,1024))

    def test_copy_accounting_includes_all_completed_vision_requests_and_rejects_drift(self):
        module=self.finder()
        for drift in (None,"generated_tokens","accepted_tokens"):
            with self.subTest(drift=drift),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);image=root/"image.png";image.write_bytes(b"fixture")
                corpus=root/"corpus";corpus.write_text("fixture")
                args=module.arguments(["--model","model","--mmproj","projector","--copy-probe","--spec-mode","mtp",
                    "--context-size","8192","--decode-tokens","8","--probe-mode","preflight","--prefill-text",str(corpus)])
                args.port=0;args.output=root
                args._vision_bootstrap=[{"ratio":r,"image":str(image),"shape":{}} for r in module.VISION_RATIOS]
                def request(url,body,**kwargs):
                    if url.endswith("/tokenize"):return {"tokens":[7] if body["content"]=="B" else list(range(2048))}
                    proof="logit_bias" in body;vision=isinstance(body["prompt"],dict)
                    generated,accepted=(11,8) if proof else (5,3) if vision else (3,2)
                    return {"tokens":[7 if proof else 42]*body["n_predict"],"timings":{"prompt_n":64 if vision else len(body["prompt"]),
                        "predicted_n":body["n_predict"],"draft_n":generated,"draft_n_accepted":accepted}}
                def launch(command,**kwargs):
                    kwargs["stdout"].write("completed fixture\n");kwargs["stdout"].flush();return object()
                stats={"draft-mtp":{"generated_tokens":49,"accepted_tokens":31}}
                if drift:stats["draft-mtp"][drift]+=1
                with patch.object(module.subprocess,"Popen",side_effect=launch),patch.object(module,"request_json",side_effect=request), \
                        patch.object(module.SWEEP,"wait_ready"),patch.object(module.SWEEP,"stop_process"), \
                        patch.object(module.SWEEP,"validate_execution_log"),patch.object(module,"check_vision_phases") as phases, \
                        patch.object(module,"speculative_source_stats",return_value=stats):
                    if drift:
                        with self.assertRaisesRegex(RuntimeError,"source totals"):module._runtime_probe(args,2048)
                    else:
                        try:result=module._runtime_probe(args,2048)
                        except RuntimeError as error:self.fail("valid text, proof and seven-image totals were rejected: "+str(error))
                        self.assertTrue(result["fits"])
                        self.assertEqual(len(result["vision_bootstrap"]),7)
                    self.assertEqual(len(phases.call_args.args[1]),7)

    def test_failed_serial_common_setting_lowers_the_published_budget(self):
        module,plan=self.plan();plan["effective_image_max_tokens"]=66
        for row in plan["ratios"]:
            row["budget_shape_ids"]=[1,1,1];row["candidates"][0]["image_max_tokens"]=66
        args=module.arguments(["--model","model","--mmproj","projector"])
        def sequence(config,arena,report,budget,directory):return {"fits":budget<66}
        with tempfile.TemporaryDirectory() as directory:
            args.output=Path(directory)
            with patch.object(module,"probe",return_value={"fits":True}),patch.object(module,"vision_common_sequence",side_effect=sequence):
                result=module.vision_capacity(args,1024,plan)
            self.assertEqual(result["common_image_max_tokens"],65)
            row=result["ratios"][0]["runtime_attempts"][0]
            self.assertEqual(row["image_sha256"],module.hashlib.sha256((Path(row["record"]).parent/"input.png").read_bytes()).hexdigest())

    def test_serial_vision_phase_validation_requires_every_completed_shape(self):
        module,plan=self.plan();shape=plan["ratios"][0]["candidates"][0]
        line="vision_phase: parent=1073741824 image_width=256 image_height=256 image_tokens=64 weights=100 compute=200 graph=150 operator=25 library=20 padding=5 borrowed_after=0 suspended_after=0 diagnostics=1"
        self.assertEqual(len(module.check_vision_phases(line+"\n"+line,[shape,shape],1024)),2)
        with self.assertRaises(RuntimeError):module.check_vision_phases(line,[shape,shape],1024)
        with self.assertRaises(RuntimeError):module.check_vision_phases(line+"\n"+line.replace("image_height=256","image_height=128"),[shape,shape],1024)

    def test_capacity_png_is_bounded_without_image_dependencies(self):
        module=self.finder()
        self.assertTrue(module.vision_png(256,128).startswith(b"\x89PNG"))
        for size in [(0,1),(65537,1),(65536,65536)]:
            with self.assertRaises(ValueError):module.vision_png(*size)


if __name__ == '__main__':
    unittest.main()
