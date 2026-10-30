"""Read-only, expensive observations for isolated diagnostics, never production hooks."""

import math
from contextlib import contextmanager
from functools import wraps

import torch
from torch import nn
from torch.nn import functional as F

from .diagnostic_protocol import DiagnosticError
from .diagnostic_failures import DiagnosticStageFailure, diagnostic_stage, failure_details


def _observation(stage):
    def decorate(method):
        @wraps(method)
        def call(self, *args, **kwargs):
            try:
                return method(self, *args, **kwargs)
            except DiagnosticStageFailure:
                raise
            except Exception as exc:
                raise DiagnosticStageFailure(failure_details(exc, origin="observation", stage=stage,
                    epoch_index=self.epoch_index, batch_index=self.batch_index)) from exc
        return call
    return decorate


def tensor_summary(tensor):
    if tensor is None:
        return None
    value = tensor.detach().to(device="cpu", dtype=torch.float64, copy=True).reshape(-1)
    finite = torch.isfinite(value)
    count = int(finite.sum().item())
    result = {"elements": value.numel(), "finite_elements": count, "nonfinite_elements": value.numel() - count,
              "min": None, "max": None, "mean": None, "l2": None, "quantiles_0_50_90_99_100": None}
    if count == value.numel() and count:
        result.update(min=float(value.min()), max=float(value.max()), mean=float(value.mean()),
                      l2=float(torch.linalg.vector_norm(value)),
                      quantiles_0_50_90_99_100=torch.quantile(value, torch.tensor(
                          [0., .5, .9, .99, 1.], dtype=torch.float64, device="cpu")).tolist())
    # Overflow in a descriptive reduction is missing evidence, never JSON Infinity.
    for key in ("min", "max", "mean", "l2"):
        if result[key] is not None and not math.isfinite(result[key]):
            result[key] = None
    return result


def optimizer_summary(model, optimizer):
    return {name: {key: tensor_summary(state[key]) for key in ("square_avg", "momentum_buffer", "grad_avg", "step")
                   if isinstance(state.get(key), torch.Tensor)}
            for name, parameter in model.named_parameters()
            for state in (optimizer.state.get(parameter, {}),)}


def bn_summary(model):
    return {name: {"momentum": module.momentum, "epsilon": module.eps,
                   "running_mean": tensor_summary(module.running_mean),
                   "running_var": tensor_summary(module.running_var),
                   "num_batches_tracked": None if module.num_batches_tracked is None else int(module.num_batches_tracked.item()),
                   "affine_weight": tensor_summary(module.weight), "affine_bias": tensor_summary(module.bias)}
            for name, module in model.named_modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)}


class BatchObserver:
    """No extra model forwards/backwards, random sampling, state writes or decisions.

    CPU-owned copies support actual update/weight ratios. Unreduced loss is a
    detached duplicate calculation; gradients still come only from the real loss.
    Hooks return None. Disabled/unsampled batches retain no tensors or hooks work.
    """
    def __init__(self, model, optimizer, *, every, activation_modules, emit):
        self.model, self.optimizer, self.every, self.emit = model, optimizer, every, emit
        self.row, self.before, self.handles = None, {}, []
        self.epoch_index, self.batch_index = None, None
        self.observation_errors = []
        modules = dict(model.named_modules())
        names = ([name for name, module in modules.items() if isinstance(module, (nn.Conv2d, nn.Linear, nn.modules.batchnorm._BatchNorm))]
                 if activation_modules == ["*"] else activation_modules)
        if any(name not in modules for name in names):
            raise DiagnosticError("Unknown selected activation module")
        self.activation_names = tuple(names)

    def __enter__(self):
        modules = dict(self.model.named_modules())
        try:
            for name in self.activation_names:
                def hook(_module, _inputs, output, name=name):
                    if self.row is not None:
                        with diagnostic_stage("observation", f"activation:{name or '<root>'}",
                                epoch_index=self.epoch_index, batch_index=self.batch_index):
                            values = (output,) if isinstance(output, torch.Tensor) else output
                            self.row["activations"].append({"module": name, "outputs": [tensor_summary(v) for v in values if isinstance(v, torch.Tensor)]})
                self.handles.append(modules[name].register_forward_hook(hook))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    @_observation("batch_snapshot")
    def start(self, epoch, batch, *, validation=False):
        self.epoch_index, self.batch_index = epoch, batch
        self.row = None
        self.before.clear()
        if not self.every or batch % self.every:
            return
        self.row = {"epoch_index": epoch, "batch_index": batch, "split": "clean_validation" if validation else "training", "activations": [],
                    "observation_boundary": "forward_before_evaluator_loss_guard" if validation else "training_forward_backward_update",
                    "lr": [float(group["lr"]) for group in self.optimizer.param_groups],
                    "bn_before_forward": bn_summary(self.model)}
        if not validation:
            self.row["optimizer_before"] = optimizer_summary(self.model, self.optimizer)
            self.before = {name: parameter.detach().to("cpu", copy=True) for name, parameter in self.model.named_parameters()}

    @_observation("forward_observation")
    def forward(self, logits, labels, loss):
        if self.row is None:
            return
        with torch.no_grad(), torch.autocast(device_type=logits.device.type, enabled=False):
            losses = F.cross_entropy(logits.detach().float(), labels, reduction="none", label_smoothing=0.)
            self.row.update(logits=tensor_summary(logits), per_example_loss=tensor_summary(losses),
                actual_mean_loss=tensor_summary(loss),
                prediction_histogram=torch.bincount(logits.detach().argmax(1), minlength=logits.shape[1]).cpu().tolist(),
                bn_after_forward=bn_summary(self.model))

    @_observation("gradient_observation")
    def gradients(self):
        if self.row is not None:
            self.row["gradients"] = {name: tensor_summary(p.grad) for name, p in self.model.named_parameters()}
            self.row["parameters_before_update"] = {name: tensor_summary(p) for name, p in self.model.named_parameters()}

    @_observation("loss_observation")
    def mean_loss(self, loss):
        if self.row is not None:
            self.row["actual_mean_loss"] = tensor_summary(loss)

    @_observation("validation_observation")
    def finish_readonly(self, failure=None):
        if self.row is not None:
            self.row["failure"] = failure
            try:
                self.emit(self.row)
            finally:
                self.row = None
                self.before.clear()

    @contextmanager
    def validation_runtime(self, runtime, epoch, *, observe=True):
        """Delegate the original evaluator's batch/precision path unchanged.

        A temporary root hook observes its actual forward using those same labels;
        it introduces no second model pass or changes to buffers/modes. The real
        evaluator still computes its own loss and complete split metrics.
        """
        if not self.every or not observe:
            yield runtime
            return
        observer = self
        class RuntimeProxy:
            counter = 0
            labels = None

            def __getattr__(self, name):
                return getattr(runtime, name)

            def batch(self, batch, classes):
                images, self.labels = runtime.batch(batch, classes)
                observer.start(epoch, self.counter, validation=True)
                self.counter += 1
                return images, self.labels

        proxy = RuntimeProxy()
        def observe(_module, _inputs, logits):
            observer.forward(logits, proxy.labels, None)
            observer.finish_readonly()
        with diagnostic_stage("observation", "validation_hook_setup", epoch_index=epoch):
            handle = self.model.register_forward_hook(observe)
        try:
            yield proxy
        except BaseException as exc:
            try:
                self.finish_readonly(failure=failure_details(exc, origin="model_computation", stage="clean_validation",
                    epoch_index=epoch, batch_index=self.batch_index))
            except Exception as trace_error:
                self.observation_errors.append(failure_details(trace_error, origin="observation", stage="failure_trace",
                    epoch_index=epoch, batch_index=self.batch_index))
            raise
        finally:
            with diagnostic_stage("observation", "validation_hook_cleanup", epoch_index=epoch, batch_index=self.batch_index):
                handle.remove()
            self.row = None
            self.before.clear()

    @_observation("update_observation")
    def finish(self, *, failure=None):
        if self.row is None:
            return
        updates = {}
        for name, parameter in self.model.named_parameters():
            previous = self.before[name].double()
            delta = parameter.detach().to("cpu", dtype=torch.float64, copy=True) - previous
            magnitude, scale = tensor_summary(delta), tensor_summary(previous)
            ratio = (None if magnitude["l2"] is None or scale["l2"] in (None, 0.)
                     else magnitude["l2"] / scale["l2"])
            updates[name] = {"actual_update": magnitude, "weight": scale,
                             "update_to_weight_l2": ratio if ratio is None or math.isfinite(ratio) else None,
                             "zero_weight_denominator": scale["l2"] == 0.}
        self.row.update(updates=updates, optimizer_after=optimizer_summary(self.model, self.optimizer),
                        bn_after_update=bn_summary(self.model), failure=failure)
        try:
            self.emit(self.row)
        finally:
            self.row = None
            self.before.clear()

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.before.clear()
        self.row = None
