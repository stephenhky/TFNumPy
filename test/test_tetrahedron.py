
import unittest
import numpy as np

import tfnumpy


class test_tetrahedron(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_embedding(self):
        tetrahedron_points = [np.array([0., 0., 0.]), np.array([1., 0., 0.]),
                              np.array([np.cos(np.pi / 3), np.sin(np.pi / 3), 0.]),
                              np.array([0.5, 0.5 / np.sqrt(3), np.sqrt(2. / 3.)])]

        sampled_points = np.concatenate(
            [np.random.multivariate_normal(point, np.eye(3) * 0.0001, 10) for point in tetrahedron_points])

        init_points = np.concatenate(
            [np.random.multivariate_normal(point[:2], np.eye(2) * 0.0001, 10) for point in tetrahedron_points])

        N = sampled_points.shape[0]
        d = sampled_points.shape[1]

        embed_pts = tfnumpy.embed.sammon_embedding(sampled_points, init_points)

        self.assertEqual(embed_pts.shape[1], 2)
        self.assertEqual(embed_pts.shape[0], N)


if __name__ == '__main__':
    unittest.main()
