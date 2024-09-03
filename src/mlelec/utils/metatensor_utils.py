import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

# def _to_tensormap(block_data: Dict):
#     # support for blocks in dict format- just convert to tensormap before using tensormap utils
#     pass


class TensorBuilder:
    def __init__(
        self, key_names, sample_names, component_names, property_names, device="cpu"
    ):
        self._key_names = key_names
        self.blocks = {}

        self._sample_names = sample_names
        self._component_names = component_names
        self._property_names = property_names

        self.device = device

    def add_block(
        self, key, gradient_samples=None, *, samples=None, components, properties=None
    ):
        if samples is None and properties is None:
            raise Exception("can not have both samples & properties unset")

        if samples is not None and properties is not None:
            raise Exception("can not have both samples & properties set")

        if samples is not None:
            if isinstance(samples, torch.ScriptObject):
                if samples._type().name() == "Labels":
                    samples = samples.values.reshape(samples.shape[0], -1)
            samples = Labels(self._sample_names, samples)

        if gradient_samples is not None:
            if not isinstance(gradient_samples, torch.ScriptObject):
                if gradient_samples._type().name() == "Labels":
                    raise Exception("must pass gradient samples for the moment")

        # print(components)
        if all([isinstance(component, torch.ScriptObject) for component in components]):
            # if all([isinstance(component, Labels) for component in components]):
            components = [
                component.values.reshape(components.shape[0], -1)
                for component in components
            ]

        components_label = []
        for names, values in zip(self._component_names, components):
            components_label.append(Labels(names, values))
        components = components_label

        if properties is not None:
            if isinstance(properties, torch.ScriptObject):
                if properties._type().name() == "Labels":
                    properties = properties.view(dtype=torch.int32).reshape(
                        properties.shape[0], -1
                    )
            elif isinstance(properties, np.ndarray):
                properties = torch.from_numpy(properties)
            elif isinstance(properties, list):
                properties = torch.tensor(list)

            properties = Labels(self._property_names, properties)

        if properties is not None:
            block = TensorBuilderPerSamples(
                properties,
                components,
                self._sample_names,
                gradient_samples,
                device=self.device,
            )

        if samples is not None:
            block = TensorBuilderPerProperties(
                samples,
                components,
                self._property_names,
                gradient_samples,
                device=self.device,
            )

        self.blocks[key] = block
        return block

    def build(self):
        keys = Labels(
            self._key_names, torch.tensor(list(self.blocks.keys()), dtype=torch.int32)
        ).to(device=self.device)

        blocks = []
        for block in self.blocks.values():
            if isinstance(block, torch.ScriptObject):
                if block._type().name() == "TensorBlock":
                    blocks.append(block)
            elif isinstance(block, TensorBuilderPerProperties):
                blocks.append(block.build())
            elif isinstance(block, TensorBuilderPerSamples):
                blocks.append(block.build())
            else:
                Exception("Invalid block type")

        self.blocks = {}
        return TensorMap(keys, blocks)


class TensorBuilderPerSamples:
    def __init__(
        self, properties, components, sample_names, gradient_samples=None, device="cpu"
    ):
        assert (
            isinstance(properties, torch.ScriptObject)
            and properties._type().name() == "Labels"
        )
        assert all(
            [
                (
                    isinstance(component, torch.ScriptObject)
                    and component._type().name() == "Labels"
                )
                for component in components
            ]
        )
        assert (gradient_samples is None) or (
            isinstance(gradient_samples, torch.ScriptObject)
            and gradient_samples._type().name() == "Labels"
        )

        self._gradient_samples = gradient_samples
        self._properties = properties
        self._components = components

        self._sample_names = sample_names
        self._samples = []

        self._data = []
        self._gradient_data = []
        self.device = device

    def add_samples(self, labels, data, gradient=None):
        if not isinstance(data, torch.Tensor):
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data).to(device=self.device)
            assert isinstance(
                data, torch.Tensor
            ), "Data must be numpy.ndarray or torch.tensor."
        assert (
            data.shape[-1] == self._properties.values.shape[0]
        ), "The property dimension of data does not match."

        for i in range(len(self._components)):
            assert (
                data.shape[i + 1] == self._components[i].values.shape[0]
            ), f"The {i}-th component dimension of data does not match."

        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).to(dtype=torch.int32, device=self.device)
        elif isinstance(labels, list):
            labels = torch.tensor(labels, dtype=torch.int32, device=self.device)

        if len(data.shape) == 2:
            data = data.reshape(1, data.shape[0], data.shape[1])
        assert data.shape[0] == labels.shape[0], (
            "data.shape[0]",
            data.shape[0],
            "labelsshape",
            labels.shape[0],
        )

        self._samples.append(labels)
        self._data.append(data.to(device=self.device))

        if gradient is not None:
            raise (Exception("Gradient data not implemented for BlockBuilderSamples"))

    def build(self):
        samples = Labels(self._sample_names, torch.vstack(self._samples))
        block = TensorBlock(
            values=torch.cat(self._data, axis=0).to(device=self.device),
            samples=samples.to(device=self.device),
            components=[c.to(device=self.device) for c in self._components],
            properties=self._properties.to(device=self.device),
        )

        if self._gradient_samples is not None:
            raise (Exception("Gradient data not implemented for BlockBuilderSamples"))

        self._gradient_data = []
        self._data = []
        self._properties = []

        return block


class TensorBuilderPerProperties:
    def __init__(
        self, samples, components, property_names, gradient_samples=None, device="cpu"
    ):
        assert isinstance(samples, torch.ScriptObject)
        # assert isinstance(samples, Labels)
        assert all(
            [isinstance(component, torch.ScriptObject) for component in components]
        )
        # assert all([isinstance(component, Labels) for component in components])
        assert (gradient_samples is None) or isinstance(
            gradient_samples, torch.ScriptObject
        )
        # assert (gradient_samples is None) or isinstance(gradient_samples, Labels)
        self._gradient_samples = gradient_samples
        self._samples = samples
        self._components = components

        self._property_names = property_names
        self._properties = []

        self._data = []
        self._gradient_data = []

        self.device = device

    def add_properties(self, labels, data, gradient=None):
        if not isinstance(data, torch.Tensor):
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            elif isinstance(data, list):
                data = torch.tensor(data)
            assert isinstance(data, torch.Tensor)

        assert data.shape[0] == self._samples.shape[0]
        for i in range(len(self._components)):
            assert data.shape[i + 1] == self._components[i].shape[0]

        labels = np.array(labels)
        if len(data.shape) == 2:
            data = data.reshape(data.shape[0], data.shape[1], 1)
        assert data.shape[2] == labels.shape[0]

        self._properties.append(labels)
        self._data.append(data)

        if gradient is not None:
            if len(gradient.shape) == 2:
                gradient = gradient.reshape(gradient.shape[0], gradient.shape[1], 1)

            assert gradient.shape[2] == labels.shape[0]
            self._gradient_data.append(gradient)

    def build(self):
        properties = Labels(self._property_names, torch.vstack(self._properties))
        block = TensorBlock(
            values=torch.cat(self._data, dim=2).to(device=self.device),
            samples=self._samples.to(device=self.device),
            components=[c.to(device=self.device) for c in self._components],
            properties=properties.to(device=self.device),
        )

        if self._gradient_samples is not None:
            block.add_gradient(
                "positions",
                self._gradient_samples,
                torch.cat(self._gradient_data, dim=2),
            )

        self._gradient_data = []
        self._data = []
        self._properties = []

        return block


def labels_where(labels, selection, return_idx=False):
    # TODO: slow! avoid using it
    # Extract the relevant columns from `selection` that the selection will
    # be performed on
    keys_out_vals = [[k[name] for name in selection.names] for k in labels]

    # First check that all of the selected keys exist in the output keys
    for slct in selection.values:
        if not torch.any([torch.all(slct == k) for k in keys_out_vals]):
            raise ValueError(
                f"selected key {selection.names} = {slct} not found"
                " in the output keys. Check the `selection` argument."
            )

    # Build a mask of the selected keys
    mask = [
        torch.any([torch.all(i == j) for j in selection.values]) for i in keys_out_vals
    ]

    labels = Labels(names=labels.names, values=labels.values[mask])
    if return_idx:
        return labels, torch.where(mask)[0]
    return labels
