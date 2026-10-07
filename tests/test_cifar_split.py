import unittest

from src.data.cifar import make_cifar_split_indices


class CifarSplitTests(unittest.TestCase):
    def test_cifar_split_has_required_sizes_and_no_overlap(self) -> None:
        training, validation = make_cifar_split_indices(split_seed=0)
        self.assertEqual(len(training), 45_000)
        self.assertEqual(len(validation), 5_000)
        self.assertTrue(set(training).isdisjoint(validation))
        self.assertEqual(len(set(training) | set(validation)), 50_000)

    def test_validation_split_is_reproducible(self) -> None:
        first_training, first_validation = make_cifar_split_indices(split_seed=123)
        second_training, second_validation = make_cifar_split_indices(split_seed=123)
        self.assertEqual(first_training, second_training)
        self.assertEqual(first_validation, second_validation)

    def test_validation_split_depends_only_on_explicit_split_seed(self) -> None:
        # There is deliberately no model/training seed argument to this function.
        _, validation = make_cifar_split_indices(split_seed=0)
        _, repeated_validation = make_cifar_split_indices(split_seed=0)
        self.assertEqual(validation, repeated_validation)
