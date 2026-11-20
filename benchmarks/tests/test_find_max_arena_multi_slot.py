from pathlib import Path
import importlib.util, unittest, tempfile, json, urllib.request
from unittest.mock import patch
root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('ms12_finder', root / 'benchmarks/find-max-arena.py')
assert spec and spec.loader
finder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finder)

class MultiSlotFinder(unittest.TestCase):

    def args(self, *extra):
        return finder.arguments(['--model', 'model.gguf', '--context-size', '8192', *extra])

    def test_two_slots_keep_context_per_slot_and_selected_group(self):
        args = self.args('--parallel', '2')
        command = finder.server_command(args, 1024)
        self.assertEqual(command[command.index('--parallel') + 1], '2')
        self.assertIn('--kv-unified', command)
        self.assertEqual(command[command.index('--ctx-size') + 1], '8192')
        self.assertEqual(finder.context_semantics(args), {'slots': 2, 'per_slot': 8192, 'selected_group': 8192, 'all_histories': 16384})

    def test_configured_N_probes_every_logical_id_and_a_real_joint_group(self):
        for count in (3, 4, 8, 256):
            with self.subTest(count=count):
                args = self.args('--parallel', str(count), '--context-size', '262144', '--probe-mode', 'preflight')
                command = finder.server_command(args, 1024)
                self.assertIn('--kv-unified', command)
                self.assertEqual(finder.context_semantics(args)['all_histories'], 262144 * count)
                profiles = finder.multi_slot_probe_profiles(args)
                self.assertEqual([row['slots'] for row in profiles[:count]], [[slot] for slot in range(count)])
                joint = next(row for row in profiles if row['label'] == 'joint')
                self.assertEqual(joint['slots'], list(range(count)))
                self.assertEqual(joint['selected_max'], count)
                self.assertLessEqual(sum(joint['prompt_tokens']) + count * (args.decode_tokens + 15), args.context_size)
                props = {'total_slots': count, 'stream_context_limits': dict(finder.context_semantics(args), selected_storage=262144)}
                finder.validate_context_semantics(args, props)
                props['stream_context_limits']['all_histories'] -= 1
                with self.assertRaisesRegex(RuntimeError, 'context'):
                    finder.validate_context_semantics(args, props)

    def test_joint_probes_respect_complete_native_verification_blocks(self):
        args = self.args('--parallel', '64', '--context-size', '262144', '--spec-mode', 'dflash',
                         '--model-draft', 'draft.gguf', '--batch-size', '16', '--ubatch-size', '16', '--probe-mode', 'preflight')
        profiles = finder.multi_slot_probe_profiles(args)
        joint = next(row for row in profiles if row['label'] == 'joint')
        self.assertEqual(joint['selected_max'], 2)
        self.assertEqual(joint['slots'], [0,1])
        self.assertEqual(len([row for row in profiles if row['label'].startswith('single-')]), 64)
        unequal = next(row for row in profiles if row['label'] == 'asymmetric')
        self.assertNotEqual(unequal['prompt_tokens'][0], unequal['prompt_tokens'][1])
        self.assertEqual(unequal['selected_max'], 2)

    def test_selected_probe_count_is_distinct_from_logical_capacity(self):
        args = self.args('--parallel','4','--probe-selected','2','--probe-mode','preflight')
        self.assertEqual(finder.context_semantics(args)['slots'],4)
        self.assertEqual(finder.context_semantics(args)['all_histories'],32768)
        profiles=finder.multi_slot_probe_profiles(args)
        self.assertEqual([row['slots'] for row in profiles[:4]],[[0],[1],[2],[3]])
        joint=next(row for row in profiles if row['label']=='joint')
        self.assertEqual(joint['slots'],[0,1])
        self.assertEqual(joint['selected_max'],2)
        for selected in ('0','5'):
            with self.assertRaises(SystemExit):
                self.args('--parallel','4','--probe-selected',selected)

    def test_rejects_counts_outside_native_sequence_contract(self):
        for count in (0, -1, 257):
            with self.subTest(count=count), self.assertRaises(SystemExit):
                self.args('--parallel', str(count))

    def test_explicit_no_warmup_is_forwarded_without_changing_probe_count(self):
        args=self.args('--parallel','4','--probe-selected','2','--no-warmup')
        self.assertIn('--no-warmup',finder.server_command(args,1024))
        self.assertNotIn('--no-warmup',finder.server_command(self.args(),1024))
        self.assertEqual(next(row for row in finder.multi_slot_probe_profiles(args) if row['label']=='joint')['selected_max'],2)

    def test_explicit_server_loads_its_own_runtime_instead_of_an_older_build(self):
        server=Path('/tmp/ms16-runtime/bin/llama-server')
        old=str(root/'build-upstream-attention-port/bin')
        with patch.object(finder.SWEEP,'server_environment',return_value={'LD_LIBRARY_PATH':old+':/usr/local/cuda/lib64',
                         'GGML_CUDA_ENABLE_UNIFIED_MEMORY':'1'}):
            env=finder.environment(server)
        paths=env['LD_LIBRARY_PATH'].split(':')
        self.assertEqual(paths[0],str(server.parent))
        self.assertNotIn(old,paths)
        self.assertIn('/usr/local/cuda/lib64',paths)
        self.assertNotIn('GGML_CUDA_ENABLE_UNIFIED_MEMORY',env)

    def test_runtime_library_hashing_handles_spaces_without_python_311(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            library=directory/'libllama runtime.so'
            library.write_bytes(b'pinned native runtime')
            maps=directory/'maps'
            maps.write_text('0-1 r-xp 0 00:00 0 '+str(library)+'\n')
            def path(value):
                return maps if value=='/proc/12345/maps' else Path(value)
            with patch.object(finder,'Path',side_effect=path), patch.object(finder.hashlib,'file_digest',None,create=True):
                rows=finder.loaded_runtime_libraries(12345)
            self.assertEqual(rows,[{'path':str(library),'sha256':finder.hashlib.sha256(library.read_bytes()).hexdigest()}])

    def test_single_slot_command_and_workload_remain_the_default(self):
        args = self.args()
        self.assertEqual(args.parallel, 1)
        self.assertEqual(finder.multi_slot_probe_profiles(args), [])
        self.assertEqual(finder.server_command(args, 1024)[finder.server_command(args, 1024).index('--parallel') + 1], '1')

    def test_bounded_probes_cover_both_single_members_and_joint_group(self):
        args = self.args('--parallel', '2', '--probe-mode', 'preflight')
        profiles = finder.multi_slot_probe_profiles(args)
        self.assertEqual([x['label'] for x in profiles], ['single-0', 'single-1', 'joint'])
        self.assertEqual([x['slots'] for x in profiles], [[0], [1], [0, 1]])
        for x in profiles:
            self.assertLessEqual(sum(x['prompt_tokens']) + len(x['slots']) * (args.decode_tokens + 15), args.context_size)

    def test_full_default_workload_adds_real_overbudget_serialized_selection(self):
        args = self.args('--parallel', '2')
        profiles = finder.multi_slot_probe_profiles(args)
        serial = next((x for x in profiles if x['label'] == 'serialized'))
        self.assertGreater(sum(serial['prompt_tokens']), args.context_size)
        self.assertEqual(serial['selected_max'], 1)
        for n in serial['prompt_tokens']:
            self.assertLessEqual(n + args.decode_tokens + 15, args.context_size)

    def test_server_context_limits_are_checked_before_publishing_capacity(self):
        args = self.args('--parallel', '2')
        props = {'total_slots': 2, 'stream_context_limits': {'per_slot': 8192, 'selected_group': 8192, 'all_histories': 16384, 'selected_storage': 8192}}
        finder.validate_context_semantics(args, props)
        props['stream_context_limits']['per_slot'] = 4096
        with self.assertRaisesRegex(RuntimeError, 'context'):
            finder.validate_context_semantics(args, props)

    def test_group_vision_reports_retired_loans_and_zero_selected_device_state(self):
        shape = {'width': 32, 'height': 32, 'image_tokens': 1, 'weight_bytes': 1024, 'compute_bytes': 768, 'workspace': {'graph_bytes': 256, 'operator_bytes': 256, 'library_bytes': 256, 'padding_bytes': 0, 'total_bytes': 768}}
        line = 'vision_group_phase: parent=1048576 weights=1024 compute=768 graph=256 operator=256 library=256 padding=0 grants_after=0 selected_state_bytes=0 image_width=32 image_height=32 image_tokens=1'
        rows = finder.check_vision_phases(line + '\n' + line, [shape, shape], 1)
        self.assertEqual(len(rows), 2)
        with self.assertRaises(RuntimeError):
            finder.check_vision_phases(line.replace('grants_after=0', 'grants_after=1'), [shape], 1)

    def test_runtime_probes_validate_every_member_and_real_selection_shape(self):
        args = self.args('--parallel', '2', '--probe-mode', 'preflight', '--decode-tokens', '8')
        with tempfile.TemporaryDirectory() as temporary:
            args.output = Path(temporary)
            log = args.output / 'server.log'
            log.write_text('')
            calls = []

            def request(url, body, **kwargs):
                prompts = body['prompt'] if isinstance(body['prompt'][0], list) else [body['prompt']]
                slots = list(range(len(prompts))) if len(prompts) == 2 else [body['id_slot']]
                with log.open('a') as output:
                    output.write('stream group bind: selected=' + str(len(prompts)) + ' rows=2 phase=decode revision=1\n')
                calls.append(body)
                rows = [{'id_slot': slot, 'tokens': list(range(8)), 'timings': {'cache_n': 0, 'prompt_n': len(prompt), 'predicted_n': 8}} for slot, prompt in zip(slots, prompts)]
                return rows if len(rows) == 2 else rows[0]
            with patch.object(finder, 'request_json', side_effect=request), patch.object(finder, 'request_completion_group', side_effect=request):
                rows = finder.multi_slot_requests(args, 'http://unused', list(range(4096)), 1024, log)
            self.assertEqual(len(calls), 3)
            self.assertEqual(sum((len(row['responses']) for row in rows)), 4)
            self.assertEqual(rows[-1]['decode_selections'], [2])

    def test_paired_image_probe_requires_two_complete_independent_responses(self):
        args = self.args('--parallel', '2', '--mmproj', 'vision.gguf', '--decode-tokens', '8')
        row = {'tokens': list(range(8)), 'timings': {'cache_n': 0, 'prompt_n': 65, 'predicted_n': 8}}
        with patch.object(finder, 'request_completion_group', return_value=[dict(row, id_slot=0), dict(row, id_slot=1)]) as request:
            self.assertEqual(len(finder.image_slot_requests(args, 'http://unused', 'Describe ' + finder.MARKER, ['encoded'])), 2)
            self.assertEqual(len(request.call_args.args[1]['prompt']), 2)
        with patch.object(finder, 'request_completion_group', return_value=[dict(row, id_slot=0)]):
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                finder.image_slot_requests(args, 'http://unused', finder.MARKER, ['encoded'])

    def test_batched_completion_transport_accepts_arrays_without_weakening_object_requests(self):
        with patch.object(urllib.request, 'urlopen') as opened:
            opened.return_value.__enter__.return_value.read.return_value = json.dumps([{'id_slot': 0}, {'id_slot': 1}]).encode()
            self.assertEqual(finder.request_completion_group('http://unused', {'prompt': [[1], [2]]}, timeout=5), [{'id_slot': 0}, {'id_slot': 1}])
            opened.return_value.__enter__.return_value.read.return_value = b'[{},null]'
            with self.assertRaises(RuntimeError):
                finder.request_completion_group('http://unused', {'prompt': [[1], [2]]}, timeout=5)

    def test_group_counter_barriers_keep_exact_totals_and_copy_has_no_neural_forward(self):
        zero = {'generated_tokens': 0, 'accepted_tokens': 0, 'accept_calls': 0, 'prediction_forwards': 0}
        completed = [{'label': 'image-miss', 'timings': {'draft_n': 7, 'draft_n_accepted': 3}}, {'label': 'image-copy', 'timings': {'draft_n': 7, 'draft_n_accepted': 7}}]
        groups = [{'label': 'image-miss', 'members': 1, 'timings': completed[0]['timings'], 'source_stats': {'draft-dflash': dict(zero, generated_tokens=7, accepted_tokens=3, accept_calls=1, prediction_forwards=1), 'ngram-map-k4v': dict(zero)}}, {'label': 'image-copy', 'members': 1, 'timings': completed[1]['timings'], 'source_stats': {'draft-dflash': dict(zero), 'ngram-map-k4v': dict(zero, generated_tokens=7, accepted_tokens=7, accept_calls=1)}}]
        with patch.object(finder, 'validate_mixed_sources', return_value={'draft-dflash': {}, 'ngram-map-k4v': {}}):
            finder.validate_completed_image_sources('', completed, groups=groups)
            groups[1]['source_stats']['draft-dflash']['prediction_forwards'] = 1
            with self.assertRaises(RuntimeError):
                finder.validate_completed_image_sources('', completed, groups=groups)
if __name__ == '__main__':
    unittest.main()
