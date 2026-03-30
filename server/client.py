from core.http_env_client import HTTPEnvClient
from core.types import StepResult
from .models import SCAct, SCObs, SCSt

class SCClient(HTTPEnvClient[SCAct, SCObs]):
    def _step_payload(self, a: SCAct) -> dict:
        return {"t_id": a.t_id, "p_qty": a.p_qty}
    
    def _parse_result(self, p: dict) -> StepResult:
        o = SCObs(
            wh_inv=p['observation']['wh_inv'],
            t_cap=p['observation']['t_cap'],
            p_ord=p['observation']['p_ord'],
            d=p['done'],
            r=p['reward']
        )
        return StepResult(observation=o, reward=p['reward'], done=p['done'])
    
    def _parse_state(self, p: dict) -> SCSt:
        return SCSt(e_id=p['e_id'], s_cnt=p['s_cnt'])