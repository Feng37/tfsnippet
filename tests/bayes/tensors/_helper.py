# -*- coding: utf-8 -*-


class DerivedTensorTestMixin(object):

    tensor_class = None
    distrib_class = None
    simple_args = None

    def test_construction(self):
        dist = self.tensor_class(**self.simple_args)
        self.assertIsInstance(dist.distribution, self.distrib_class)
        # assert the distribution scope is contained in the scope
        # of StochasticTensor
        self.assertEqual(
            dist.distribution.variable_scope.name,
            '%s/distribution' % dist.variable_scope.name
        )
