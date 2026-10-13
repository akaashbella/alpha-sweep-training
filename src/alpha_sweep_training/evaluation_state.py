"""Owned exact whole-model preservation checks for independent evaluation."""

from contextlib import contextmanager
import torch

from .component_contracts import ComponentError


class CleanStatePreservationError(ComponentError):
    pass


class CleanStateBaseline:
    """All parameters and buffers, including nonpersistent ones and None bindings.

    CPU copies own their storage. Verify at realization entry/exit; on a violated
    invariant restore original bindings/values before raising. A repaired failure
    can never be published as a successful pass. Modes are restored individually.
    """
    def __init__(self, model):
        self.model = model
        self.modules = tuple(model.named_modules())
        self.module_bindings = tuple((module, dict(module._modules)) for _, module in self.modules)
        self.entries = []
        for _, module in self.modules:
            for attribute in ("_parameters", "_buffers"):
                bindings = dict(getattr(module, attribute))
                values = {name: None if tensor is None else tensor.detach().to("cpu", copy=True)
                          for name, tensor in bindings.items()}
                descriptors = {name: None if tensor is None else (tensor.device, tensor.dtype, tensor.requires_grad)
                               for name, tensor in bindings.items()}
                self.entries.append((module, attribute, bindings, values, descriptors))

    def verify(self):
        if tuple(self.model.named_modules()) != self.modules:
            raise CleanStatePreservationError("Evaluation changed module ownership/structure")
        for module, bindings in self.module_bindings:
            if set(module._modules) != set(bindings) or any(module._modules[name] is not child for name, child in bindings.items()):
                raise CleanStatePreservationError("Evaluation changed original module bindings")
        for module, attribute, bindings, values, descriptors in self.entries:
            current = getattr(module, attribute)
            if set(current) != set(bindings) or any(current[name] is not tensor for name, tensor in bindings.items()):
                raise CleanStatePreservationError("Evaluation changed original parameter/buffer bindings")
            for name, tensor in bindings.items():
                if tensor is not None and ((tensor.device, tensor.dtype, tensor.requires_grad) != descriptors[name]
                        or not torch.equal(tensor.detach().cpu(), values[name])):
                    raise CleanStatePreservationError(f"Evaluation changed clean {attribute}.{name}")

    def restore(self):
        with torch.no_grad():
            for module, bindings in self.module_bindings:
                module._modules.clear()
                module._modules.update(bindings)
            for module, attribute, bindings, values, descriptors in self.entries:
                target = getattr(module, attribute)
                target.clear()
                target.update(bindings)
                for name, tensor in bindings.items():
                    if tensor is not None:
                        if (tensor.device, tensor.dtype) != descriptors[name][:2]:
                            raise CleanStatePreservationError("Cannot repair a moved/retyped clean binding; discard evaluation model")
                        tensor.copy_(values[name])
                        tensor.requires_grad_(descriptors[name][2])

    @contextmanager
    def realization(self):
        modes = tuple((module, module.training) for _, module in self.modules)
        try:
            self.verify()
            yield
        finally:
            try:
                self.verify()
            except BaseException:
                self.restore()
                self.verify()
                raise
            finally:
                for module, training in modes:
                    module.training = training
                if any(module.training is not training for module, training in modes):
                    raise CleanStatePreservationError("Evaluation did not restore prior module modes")
