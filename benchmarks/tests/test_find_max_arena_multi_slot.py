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
