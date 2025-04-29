import uam.wrappers.MinAtar_wrapper as Min
import uam.wrappers.gym_POMDP_wrapper as gw

def get_wrapper(*, name: str, **kwargs) -> type:

    if "MinAtar" in name:
        return Min.MinAtar_wrapper(**kwargs)
    elif "POMDP" in name:
        return gw.gym_POMDP_wrapper(**kwargs)
