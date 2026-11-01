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
    def test_finder_admits_complete_pure_dflash_images_but_not_mixed_vision(self):
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
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                finder.arguments(base+["--spec-mode","dflash-k4v"])
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


if __name__ == '__main__':
    unittest.main()
