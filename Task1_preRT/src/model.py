def get_network(parser, config):
    if (
        config.get("anisotropic_scales", False)
        and "SegResNetDS" in config["network"]["_target_"]
    ):
        parser.config["network"]["resolution"] = config["resample_resolution"]
        parser.parse(reset=True)
        print("Using anisotripic scales", parser["network"])

    model = parser.get_parsed_content("network")

    return model
