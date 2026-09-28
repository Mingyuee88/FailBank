"""Published AEGIS (vlsa-aegis) as an in-loop baseline behind FailBank's teacher interface.

    failbank-rollout ... --teacher failbank_aegis:VlsaAegisTeacher --execute teacher

Needs ``AEGIS_ROOT`` (a patched vlsa-aegis checkout with GroundingDINO weights) and a
GLM-4.5V endpoint (``AEGIS_GLM_BASE_URL`` for a local vLLM server, or ``ZHIPUAI_API_KEY``
for the hosted API). Nothing in FailBank's core imports this package.
"""
from failbank_aegis.shield import VlsaAegisTeacher

__all__ = ["VlsaAegisTeacher"]
