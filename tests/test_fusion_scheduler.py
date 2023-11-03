import unittest
from pathlib import Path

import aind_cloud_fusion.io as io
import aind_cloud_fusion.runtime as runtime


class TestScheduler(unittest.TestCase):
    def __init__(self): 
        config_yaml_path = 'tests/test_scheduler_config.yml'
        self.node = runtime.Scheduler(config_yaml_path, dataset)
        self.node.run()

        params = io.read_config_yaml(config_yaml_path)
        self.worker_yml_path = params['runtime']['scheduler']['worker_yml_path']
        self.num_workers = params['runtime']['scheduler']['num_workers']

    def test_num_files(self):
        yaml_file_count = 0
        for file_path in Path(self.worker_yml_path).iterdir():
            if file_path.is_file() and file_path.suffix.lower() == '.yaml':
                yaml_file_count += 1
        
        self.assertTrue(yaml_file_count == self.num_workers)

    def test_no_work_overlap(self):
        all_work: list[tuple[int, int, int]] = []

        for file_path in Path(self.worker_yml_path).iterdir():
            if file_path.is_file() and file_path.suffix.lower() == '.yaml':
                params = io.read_config_yaml(str(file_path))
                worker_cells: list[tuple[int, int, int]] = params['runtime']['worker']['worker_cells']
                all_work.extend(worker_cells)

        unique_work = set(all_work)

        self.assertTrue(len(all_work) == len(unique_work))

    def tearDown(self):
        # Delete test worker ymls. 
        shutil.rmtree(self.worker_yml_path)


if __name__ == "__main__":
    unittest.main()