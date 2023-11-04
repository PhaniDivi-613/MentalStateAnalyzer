# def create_instance(config, asdict=False):
#     """create machine learing model by config"""
#     from .common import CONFIG_TO_CLASS
#     class_name = CONFIG_TO_CLASS[config.__class__.__name__]
#     if asdict:
#         return class_name(**config.asdict())
#     else:
#         return class_name(config)