from .coma import COMACritic
from .centralV import CentralVCritic
from .coma_ns import COMACriticNS
from .centralV_ns import CentralVCriticNS
from .maddpg import MADDPGCritic
from .maddpg_ns import MADDPGCriticNS
from .ac import ACCritic
from .ac_ns import ACCriticNS
from .mat import MATCritic


REGISTRY = {}

REGISTRY["coma_critic"] = COMACritic
REGISTRY["cv_critic"] = CentralVCritic
REGISTRY["coma_critic_ns"] = COMACriticNS
REGISTRY["cv_critic_ns"] = CentralVCriticNS
REGISTRY["maddpg_critic"] = MADDPGCritic
REGISTRY["maddpg_critic_ns"] = MADDPGCriticNS
REGISTRY["ac_critic"] = ACCritic
REGISTRY["ac_critic_ns"] = ACCriticNS
REGISTRY["mat_critic"] = MATCritic


def register_pac_critics():
    """Lazy registration of PAC critics.

    PAC critics depend on torch_scatter (pac_dcg_ns) which is a heavy
    optional dependency (requires matching torch build). Keeping them out
    of the eager import path lets qmix/iql/vdn/coma run without installing
    torch_scatter. Call this from the PAC learners' __init__.
    """
    from .pac_ac_ns import PACCriticNS
    from .pac_dcg_ns import DCGCriticNS

    REGISTRY["pac_critic_ns"] = PACCriticNS
    REGISTRY["pac_dcg_critic_ns"] = DCGCriticNS
