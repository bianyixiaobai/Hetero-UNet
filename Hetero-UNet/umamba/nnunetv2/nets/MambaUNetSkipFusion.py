"""D (baseline + skip fusion) and E (frozen C + skip fusion) factories."""
from nnunetv2.nets.multiscale_skip_fusion import MultiScaleSkipDecoder


def get_skip_fusion_mamba_from_plans(plans_manager, dataset_json, configuration_manager,
                                    num_input_channels, deep_supervision=True,
                                    with_bg=False, fusion_width=16):
    if with_bg:
        from nnunetv2.nets.MambaUNetAttnRes import get_attnres_mamba_from_plans
        model = get_attnres_mamba_from_plans(
            plans_manager, dataset_json, configuration_manager, num_input_channels,
            deep_supervision=deep_supervision, block_size=3,
            boundary_guidance=True, auxiliary_boundary=True, boundary_hidden_channels=8)
    else:
        from nnunetv2.nets.MambaUNet import get_umamba_bot_from_plans
        model = get_umamba_bot_from_plans(
            plans_manager, dataset_json, configuration_manager, num_input_channels,
            deep_supervision=deep_supervision)
    # The base has already been initialized. Never reinitialize the complete
    # network: that would destroy frozen C query initialization and zero output.
    model.decoder = MultiScaleSkipDecoder(
        model.decoder, len(configuration_manager.conv_kernel_sizes[0]), fusion_width)
    return model
