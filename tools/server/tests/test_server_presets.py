import unittest
from unittest.mock import Mock, patch

import utils


class PresetProvisioning(unittest.TestCase):
    def setUp(self):
        self.first, self.second = Mock(), Mock()
        first, second = self.first, self.second

        class Presets:
            @staticmethod
            def first_model():
                return first

            @staticmethod
            def second_model():
                return second

        self.load = utils.ServerPreset.load_all
        patched = patch.object(utils, "ServerPreset", Presets)
        patched.start()
        self.addCleanup(patched.stop)

    def test_provisioning_uses_extended_deadline_and_stops_each_server(self):
        self.load()
        for server in (self.first, self.second):
            self.assertFalse(server.offline)
            server.start.assert_called_once_with(timeout_seconds=utils.DEFAULT_REQUEST_TIMEOUT)
            server.stop.assert_called_once_with()

    def test_failed_provisioning_stops_the_server_and_propagates_failure(self):
        self.first.start.side_effect = TimeoutError("provisioning deadline")
        with self.assertRaisesRegex(TimeoutError, "provisioning deadline"):
            self.load()
        self.first.stop.assert_called_once_with()
        self.second.start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
