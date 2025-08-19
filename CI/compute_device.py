#!/usr/bin/env python
# ******************************************************************************
# Copyright 2025 Brainchip Holdings Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
__all__ = ["compute_min_device", "compute_common_device"]

import warnings
from collections import namedtuple

import akida


LayerSequence = namedtuple('LayerSequence', ['layers'])


def _get_outbounds(layer, layers):
    return [ly for ly in layers if layer in ly.inbounds]


def _model_generator(layers):
    # Scroll through a list of layers, returning a pair of consecutive layers. Notes:
    # - one of the two branches must not have nodes (not implemented yet)
    # - merge layer is performed in the following NP, so we take their inbounds
    queue = [layers[-1]]
    while len(queue) > 0:
        t_layer = queue.pop(0)
        inbounds = t_layer.inbounds
        # Skip a layer if it is a merge one.
        if len(inbounds) == 1 and len(inbounds[0].inbounds) > 1:
            inbounds = inbounds[0].inbounds
        # Check inbounds constraints.
        if len(inbounds) > 1:
            # In case of multiple branches, one of them must not contain layers.
            # This translates to some inbound having multiple outbounds.
            new_inbounds = []
            for ly in inbounds:
                # Remove the branch with empty layers
                if len(_get_outbounds(ly, layers)) == 1:
                    new_inbounds.append(ly)
            if len(new_inbounds) != 1:
                raise NotImplementedError(f"{t_layer} has multiple inbounds, "
                                          "but there is no empty branch.")
            # Remove the inbounds that are not empty branches.
            inbounds = new_inbounds

        # Yield the pair (inbound, target_layer) if both have been mapped
        # Or if taget_layer is mapped and the inbound is an InputData layer.
        if len(inbounds) == 1 and t_layer.mapping is not None:
            if inbounds[0].parameters.layer_type == akida.LayerType.InputData or \
                    inbounds[0].mapping is not None:
                yield LayerSequence((inbounds[0], t_layer))
                # Then, update the queue with the inbound layer.
                queue.append(inbounds[0])


def _get_initial_skip_dma_channels(model):
    # The initial number of skip DMAs is len(btc) + len(skips)
    SKIP_LAYER_TYPES = [akida.LayerType.Add, akida.LayerType.Concatenate]
    BTC_LAYER_TYPES = [akida.LayerType.BufferTempConv, akida.LayerType.DepthwiseBufferTempConv]
    skip_dma_channels = 0
    for ly in model.layers:
        if ly.parameters.layer_type in SKIP_LAYER_TYPES + BTC_LAYER_TYPES:
            skip_dma_channels += 1
    return skip_dma_channels


def _get_initial_number_of_fnp(model):
    # The initial number of FNP is len(dense), since they are not split
    FNP_LAYER_TYPES = [akida.LayerType.Dense1D]
    nb_fnp = 0
    for ly in model.layers:
        if ly.parameters.layer_type in FNP_LAYER_TYPES:
            nb_fnp += 1
    return nb_fnp


def _get_np_components(model_or_pass, np_types=None):
    total_nps = []
    for layer in model_or_pass.layers:
        if hasattr(layer.mapping, 'nps'):
            for np in layer.mapping.nps:
                if np_types is None or np.type in np_types:
                    total_nps.append(np)
        if hasattr(layer.mapping, 'skipdma_loads'):
            for np in layer.mapping.skipdma_loads:
                if np_types is None or np.type in np_types:
                    total_nps.append(np)
        if hasattr(layer.mapping, 'skipdma_stores'):
            for np in layer.mapping.skipdma_stores:
                if np_types is None or np.type in np_types:
                    total_nps.append(np)
    return total_nps


def compute_skip_dma_channels(model_or_pass):
    # Compute the number of skip DMA channels as max(len(SKIP_DMA_STORE), len(SKIP_DMA_LOAD))
    skip_dma_load = _get_np_components(model_or_pass, (akida.NP.SKIP_DMA_LOAD,))
    skip_dma_store = _get_np_components(model_or_pass, (akida.NP.SKIP_DMA_STORE,))
    return max(len(skip_dma_load), len(skip_dma_store))


def compute_number_of_cnp_tnp(model_or_pass):
    # Compute the number of CNP/TNP-B.
    CNP_TNP_B_TYPES = (akida.NP.CNP1, akida.NP.CNP2, akida.NP.TNP_B)
    total_cnps = _get_np_components(model_or_pass, CNP_TNP_B_TYPES)
    return len(total_cnps)


def compute_number_of_fnp(model_or_pass):
    # Compute the number of FNP.
    FNP_TYPES = (akida.NP.FNP2, akida.NP.FNP3)
    total_fnps = _get_np_components(model_or_pass, FNP_TYPES)
    return len(total_fnps)


def compute_min_device(model,
                       enable_hwpr=False,
                       sram_size=None,
                       minimal_memory=False,
                       initial_num_nodes=36):
    """Builds the Akida virtual device that can fit the model entirely
    with or without reconfiguration.

    Args:
        model (akida.Model): the model used to determine the device.
        enable_hwpr (bool, optional): if True, the device is computed assuming
            partial reconfiguration. Defaults to False.
        sram_size (akida.NP.SramSize, optional): Size of shared SRAM available inside the mesh.
            Ignored when `minimal_memory` is True. Defaults to None.
        minimal_memory (bool, optional): if True, computes and sets the minimal required
            inputs and weights memory for the device. Defaults to False.
        initial_num_nodes (int, optional): the initial number of nodes with which to compute
            the base device. Defaults to 36.

    Returns:
        akida.Device: the computed device
    """
    NUM_NPS_PER_NODE = 4
    if model.ip_version != akida.IpVersion.v2:
        raise ValueError("Only IpVersion.v2 models are supported. "
                         f"Current model version={model.ip_version}")

    # Create a copy of the model to avoid modifying the original one.
    model = akida.Model(layers=model.layers)

    # Compute a base device with which to compute the next parameters.
    params = {"num_skip_dma_channel": _get_initial_skip_dma_channels(model),
              "num_fnp": _get_initial_number_of_fnp(model),
              "sram_size": sram_size}

    params["num_cnp_tnp"] = NUM_NPS_PER_NODE * initial_num_nodes - params["num_fnp"]
    if params["num_cnp_tnp"] < 0:
        raise ValueError("Impossible to compute base device: "
                         f"the number of initial nodes ({initial_num_nodes}) is not enough.")
    device = akida.create_device(**params)

    # Map model with the default parameters.
    model.map(device, mode=akida.mapping.MapMode.Minimal, hw_only=True)

    # Now that the model has been mapped onto the base device,
    # we can compute the parameters to build the required device.
    if enable_hwpr:
        params["num_cnp_tnp"] = params["num_fnp"] = 0
        for layer_seq in _model_generator(model.layers):
            # Compute the number of CNP/FNP needed to map the model in multiple passes,
            # as the larger sum of 2 consecutive layers.
            params["num_cnp_tnp"] = max(params["num_cnp_tnp"], compute_number_of_cnp_tnp(layer_seq))
            params["num_fnp"] = max(params["num_fnp"], compute_number_of_fnp(layer_seq))
        # To compute the minimum number of skip DMA channels needed when partial reconfiguration
        # is allowed, we iterate the device until we find a valid one.
        for num_skip_dma_channel in range(1, params.pop("num_skip_dma_channel") + 1):
            try:
                device = akida.create_device(num_skip_dma_channel=num_skip_dma_channel, **params)
                model.map(device, mode=akida.mapping.MapMode.Minimal, hw_only=True)
                params["num_skip_dma_channel"] = num_skip_dma_channel
                break
            except Exception:
                continue
    else:
        params["num_cnp_tnp"] = compute_number_of_cnp_tnp(model)
        params["num_fnp"] = compute_number_of_fnp(model)
        params["num_skip_dma_channel"] = compute_skip_dma_channels(model)

    if minimal_memory:
        if sram_size is not None:
            warnings.warn(
                "The 'sram_size' argument will be ignored because 'minimal_memory' is set to True. "
                "The required memory will be computed automatically."
            )
        params["sram_size"] = akida.NP.SramSize(*akida.compute_minimal_memory(model))

    # Create a virtual device with the requirements.
    device = akida.create_device(**params)

    # Sanity check: map model on device.
    try:
        model.map(device, mode=akida.mapping.MapMode.Minimal, hw_only=True)
    except Exception as e:
        raise RuntimeError("It was not possible to find a device for this model. "
                           f"Reason:\n{str(e)}")
    return device


def compute_common_device(ak_models):
    """Computes a common Akida device that can run all the given models.
    Ensures all models were mapped.

    Args:
        ak_models (List[akida.Model]): A list of Akida models whose hardware
            requirements will be combined.

    Returns:
        akida.Device: A new device that can map all the given models.
    """
    if not ak_models:
        raise ValueError("The list of Akida models cannot be empty.")

    if any(model.device is None for model in ak_models):
        raise ValueError("All models must be mapped on a device.")

    # For safety, check that all models devices have the same version
    assert all(model.device.version == ak_models[0].device.version for model in ak_models), \
        "Models devices have different versions."

    include_hrc = any(model.device.mesh.has_hrc for model in ak_models)
    max_num_cnp_tnp = 0
    max_num_fnp = 0
    max_num_skip_dma_channel = 0
    sram_size = akida.NP.SramSize(0, 0)

    for model in ak_models:
        # Update params
        max_num_cnp_tnp = max(max_num_cnp_tnp, compute_number_of_cnp_tnp(model))
        max_num_fnp = max(max_num_fnp, compute_number_of_fnp(model))
        max_num_skip_dma_channel = max(max_num_skip_dma_channel, compute_skip_dma_channels(model))

        # Update Sram size
        sram_size = akida.NP.SramSize(max(sram_size.input_bytes,
                                          model.device.mesh.np_sram_size.input_bytes),
                                      max(sram_size.weight_bytes,
                                          model.device.mesh.np_sram_size.weight_bytes))

    return akida.create_device(max_num_cnp_tnp, max_num_fnp,
                               max_num_skip_dma_channel, include_hrc,
                               sram_size, ak_models[0].device.version)
