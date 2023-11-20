import numpy as np
from metatensor import Labels, TensorBlock, TensorMap
from itertools import product
import torch
from typing import List, Optional, Union
import metatensor
import metatensor.operations as operations


def _to_tensormap(block_data: dict):
    # support for blocks in dict format- just convert to tensormap before using tensormap utils
    pass


class TensorBuilder:
    def __init__(self, key_names, sample_names, component_names, property_names):
        self._key_names = key_names
        self.blocks = {}

        self._sample_names = sample_names
        self._component_names = component_names
        self._property_names = property_names

    def add_block(
        self, keys, gradient_samples=None, *, samples=None, components, properties=None
    ):
        if samples is None and properties is None:
            raise Exception("can not have both samples & properties unset")

        if samples is not None and properties is not None:
            raise Exception("can not have both samples & properties set")

        if samples is not None:
            if isinstance(samples, Labels):
                samples = np.asarray(samples.values).reshape(samples.shape[0], -1)
            samples = Labels(self._sample_names, samples)

        if gradient_samples is not None:
            if not isinstance(gradient_samples, Labels):
                raise Exception("must pass gradient samples for the moment")

        if all([isinstance(component, Labels) for component in components]):
            components = [
                np.asarray(component.values).reshape(components.shape[0], -1)
                for component in components
            ]

        components_label = []
        for names, values in zip(self._component_names, components):
            components_label.append(Labels(names, values))
        components = components_label

        if properties is not None:
            if isinstance(properties, Labels):
                properties = properties.view(dtype=np.int32).reshape(
                    properties.shape[0], -1
                )
            properties = Labels(self._property_names, properties)

        if properties is not None:
            block = TensorBuilderPerSamples(
                properties, components, self._sample_names, gradient_samples
            )

        if samples is not None:
            block = TensorBuilderPerProperties(
                samples, components, self._property_names, gradient_samples
            )

        self.blocks[keys] = block
        return block

    def build(self):
        keys = Labels(
            self._key_names,
            np.array(list(self.blocks.keys()), dtype=np.int32),
        )

        blocks = []
        for block in self.blocks.values():
            if isinstance(block, TensorBlock):
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
    def __init__(self, properties, components, sample_names, gradient_samples=None):
        assert isinstance(properties, Labels)
        assert all([isinstance(component, Labels) for component in components])
        assert (gradient_samples is None) or isinstance(gradient_samples, Labels)
        self._gradient_samples = gradient_samples
        self._properties = properties
        self._components = components

        self._sample_names = sample_names
        self._samples = []

        self._data = []
        self._gradient_data = []

    def add_samples(self, labels, data, gradient=None):
        assert isinstance(data, np.ndarray) or isinstance(data, torch.Tensor)
        assert data.shape[-1] == self._properties.values.shape[0]
        for i in range(len(self._components)):
            assert data.shape[i + 1] == self._components[i].values.shape[0]

        labels = np.asarray(labels, dtype=np.int32)

        if len(data.shape) == 2:
            data = data.reshape(1, data.shape[0], data.shape[1])
        assert data.shape[0] == labels.shape[0]

        self._samples.append(labels)
        self._data.append(data)

        if gradient is not None:
            raise (Exception("Gradient data not implemented for BlockBuilderSamples"))

    def build(self):
        samples = Labels(self._sample_names, np.vstack(self._samples))
        block = TensorBlock(
            values=torch.cat(self._data, axis=0),
            samples=samples,
            components=self._components,
            properties=self._properties,
        )

        if self._gradient_samples is not None:
            raise (Exception("Gradient data not implemented for BlockBuilderSamples"))

        self._gradient_data = []
        self._data = []
        self._properties = []

        return block


class TensorBuilderPerProperties:
    def __init__(self, samples, components, property_names, gradient_samples=None):
        assert isinstance(samples, Labels)
        assert all([isinstance(component, Labels) for component in components])
        assert (gradient_samples is None) or isinstance(gradient_samples, Labels)
        self._gradient_samples = gradient_samples
        self._samples = samples
        self._components = components

        self._property_names = property_names
        self._properties = []

        self._data = []
        self._gradient_data = []

    def add_properties(self, labels, data, gradient=None):
        assert isinstance(data, np.ndarray)
        assert data.shape[0] == self._samples.shape[0]
        for i in range(len(self._components)):
            assert data.shape[i + 1] == self._components[i].shape[0]

        labels = np.asarray(labels)
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
        properties = Labels(self._property_names, np.vstack(self._properties))
        block = TensorBlock(
            values=np.concatenate(self._data, axis=2),
            samples=self._samples,
            components=self._components,
            properties=properties,
        )

        if self._gradient_samples is not None:
            block.add_gradient(
                "positions",
                self._gradient_samples,
                np.concatenate(self._gradient_data, axis=2),
            )

        self._gradient_data = []
        self._data = []
        self._properties = []

        return block
