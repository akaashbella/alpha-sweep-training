"""Authoritative, versioned constants for the main scientific experiment."""

from copy import deepcopy

SCHEMA_VERSION = 3
SPECIFICATION_VERSION = "main-v1"
FINGERPRINT_VERSION = 3
MODEL_BUILDER_VERSION = "torchvision-cifar-builders-v1"
MODEL_INITIALIZATION_VERSION = "sha256-isolated-cpu-float32-v1"
ELIGIBLE_WEIGHT_VERSION = "conv2d-linear-owned-weights-v1"
OPTIMIZER_BUILDER_VERSION = "single-group-native-recipe-v1"
SCHEDULER_VERSION = "completed-units-linear-cosine-v1"
PINNED_TORCH_VERSION = "2.5.1"
PINNED_TORCHVISION_VERSION = "0.20.1"
PREPROCESSING_VERSION = "cifar32-zero-pad4-flip05-float32-v1"
DATA_RANDOMNESS_VERSION = "sha256-keyed-torch-cpu-v1"

DATASETS = {"cifar10": 10, "cifar100": 100}
ARCHITECTURES = (
    "resnet50",
    "mobilenetv3_large",
    "efficientnet_b0",
    "convnext_tiny",
)
ALPHAS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00)
SEED_COUNT = 3
EPOCHS = 200
BATCH_SIZE = 128
WARMUP_EPOCHS = 5
TRAIN_SIZE = 45_000
VALIDATION_SIZE = 5_000
TEST_SIZE = 10_000
TOTAL_RUNS = len(DATASETS) * len(ARCHITECTURES) * len(ALPHAS) * SEED_COUNT

# Inactive optimizer fields remain null, so there is one explicit serialization schema.
_OPTIMIZER_RECIPES = {
    "resnet50": {
        "name": "sgd", "learning_rate": 0.1, "momentum": 0.9,
        "weight_decay": 0.0005, "rmsprop_alpha": None, "epsilon": None,
        "betas": None,
    },
    "mobilenetv3_large": {
        "name": "rmsprop", "learning_rate": 0.01, "momentum": 0.9,
        "weight_decay": 0.00001, "rmsprop_alpha": 0.9, "epsilon": 0.001,
        "betas": None,
    },
    "efficientnet_b0": {
        "name": "rmsprop", "learning_rate": 0.01, "momentum": 0.9,
        "weight_decay": 0.00001, "rmsprop_alpha": 0.9, "epsilon": 0.001,
        "betas": None,
    },
    "convnext_tiny": {
        "name": "adamw", "learning_rate": 0.001, "momentum": None,
        "weight_decay": 0.05, "rmsprop_alpha": None, "epsilon": None,
        "betas": [0.9, 0.999],
    },
}


def optimizer_recipe(architecture: str) -> dict:
    """Return an independent declaration; this does not construct an optimizer."""
    return deepcopy(_OPTIMIZER_RECIPES[architecture])


def default_document(dataset: str, architecture: str) -> dict:
    """Build fixed research settings while leaving undecided settings explicit."""
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific": {
            "specification_version": SPECIFICATION_VERSION,
            "dataset": dataset,
            "architecture": architecture,
            "alpha": 0.0,
            "training_seed": None,
            "training_seeds": None,
            "data": {
                "preprocessing_version": PREPROCESSING_VERSION,
                "randomness_version": DATA_RANDOMNESS_VERSION,
                "image_size": 32,
                "train_size": TRAIN_SIZE,
                "validation_size": VALIDATION_SIZE,
                "test_size": TEST_SIZE,
                "split": {"policy": "persisted_official_train_indices", "identity": None},
                "normalization": {"mean": None, "std": None, "source": None},
                "training_transforms": ["random_crop_padding_4", "random_horizontal_flip",
                                        "to_float32_chw_uint8_div_255", "normalize"],
                "evaluation_transforms": ["to_float32_chw_uint8_div_255", "normalize"],
            },
            "model": {
                "builder_version": MODEL_BUILDER_VERSION,
                "initialization_version": MODEL_INITIALIZATION_VERSION,
                "num_classes": DATASETS[dataset], "pretrained": False,
                "stem_policy": "cifar32", "preserve_remaining_structure": True,
            },
            "optimizer": optimizer_recipe(architecture),
            "training": {
                "scheduler_version": SCHEDULER_VERSION,
                "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                "loss": "cross_entropy", "warmup_epochs": WARMUP_EPOCHS,
                "schedule": "linear_warmup_then_cosine",
                "warmup_start_factor": None, "cosine_min_lr": None,
                "scheduler_interval": None, "drop_last": None,
                "early_stopping": False, "mixup": False, "cutmix": False,
                "label_smoothing": 0.0, "gradient_clipping": False, "ema": False,
                "native_regularization": "preserve",
            },
            "precision": {
                "mode": None, "float32_matmul_precision": None,
                "allow_tf32": None, "deterministic_algorithms": None,
                "cudnn_deterministic": None, "cudnn_benchmark": None,
            },
            "perturbation": {
                "distribution": "standard_normal", "scale": "population_std",
                "std_correction": 0, "detach_scale_and_noise": True,
                "eligible_weights": ["Conv2d.weight", "Linear.weight"],
                "refresh": "every_minibatch", "simultaneous": True,
                "update_target": "clean_master_parameters", "zero_bypass": True,
                "perturb_biases": False, "perturb_normalization": False,
                "perturb_buffers": False, "batchnorm_running_updates": True,
            },
            "evaluation": {
                "clean_weights": True, "best_metric": "validation_accuracy",
                "best_tie_policy": "earliest_epoch", "test_checkpoint": "best",
                "test_every_alpha_and_seed": True,
                "best_alpha_metric": "mean_validation_accuracy_across_seeds",
                "save_best": True, "save_final": True,
                "record_epoch_metrics": True, "record_clean_layer_statistics": True,
            },
        },
        "execution": {
            "storage": {
                "output_root": "runs", "checkpoint_root": None,
                "result_root": None, "log_root": None,
                "data_root": "data", "split_file": None,
            },
            "num_workers": 8, "device": "cuda",
            "slurm": {
                "partition": "dgxh", "account": None, "gpus": 1,
                "cpus": 8, "memory_gb": 32, "wall_time": "12:00:00",
                "max_concurrent": 4,
            },
        },
    }
