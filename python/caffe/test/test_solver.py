import unittest
import tempfile
import os
import numpy as np
import six

import caffe
from test_net import simple_net_file


class TestSolver(unittest.TestCase):
    def setUp(self):
        self.num_output = 13
        net_f = simple_net_file(self.num_output)
        f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        net_f_mod = net_f
        if os.name == 'nt':
            net_f_mod = net_f_mod.replace("\\", "/")
        f.write("""net: '""" + net_f_mod + """'
        test_iter: 10 test_interval: 10 base_lr: 0.01 momentum: 0.9
        weight_decay: 0.0005 lr_policy: 'inv' gamma: 0.0001 power: 0.75
        display: 100 max_iter: 100 snapshot_after_train: false
        snapshot_prefix: "model" """)
        f.close()
        self.solver = caffe.SGDSolver(f.name)
        # also make sure get_solver runs
        caffe.get_solver(f.name)
        caffe.set_mode_cpu()
        # fill in valid labels
        self.solver.net.blobs['label'].data[...] = \
                np.random.randint(self.num_output,
                    size=self.solver.net.blobs['label'].data.shape)
        self.solver.test_nets[0].blobs['label'].data[...] = \
                np.random.randint(self.num_output,
                    size=self.solver.test_nets[0].blobs['label'].data.shape)
        os.remove(f.name)
        os.remove(net_f)

    def test_solve(self):
        self.assertEqual(self.solver.iter, 0)
        self.solver.solve()
        self.assertEqual(self.solver.iter, 100)

    def test_apply_update(self):
        net = self.solver.net
        data = net.layers[1].blobs[0].data[...]
        # Reset the weights of that layer to 0
        data[...] = 0
        net.layers[1].blobs[0].diff[...] = 1
        # Apply the update, the initial learning rate should be 0.01
        self.solver.apply_update()
        # Check that the new weights are -0.01, with a precision of 1e-7
        self.assertTrue((data - -0.01 * np.ones(data.shape)).max() < 1e-7)

    def test_net_memory(self):
        """Check that nets survive after the solver is destroyed."""

        nets = [self.solver.net] + list(self.solver.test_nets)
        self.assertEqual(len(nets), 2)
        del self.solver

        total = 0
        for net in nets:
            for ps in six.itervalues(net.params):
                for p in ps:
                    total += p.data.sum() + p.diff.sum()
            for bl in six.itervalues(net.blobs):
                total += bl.data.sum() + bl.diff.sum()

    def test_snapshot(self):
        self.solver.snapshot()
        # Check that these files exist and then remove them
        files = ['model_iter_0.caffemodel', 'model_iter_0.solverstate']
        for fn in files:
            assert os.path.isfile(fn)
            os.remove(fn)
