# Coordination Transformer

Repository root에서 다음 agent들을 순서대로 실행한다.

```bash
GPUS=0,1 bash scripts/overcooked_v3/run_wandb_agents.sh \
  cilab-overcooked/overcooked-v3-coot-population/47ubezl3 \
  cilab-overcooked/overcooked-v3-coot-population/ueiul4xd \
  cilab-overcooked/overcooked-v3-coot-pipeline/5xvncbux \
  cilab-overcooked/overcooked-v3-coot-response-candidates/gojtpiqm \
  cilab-overcooked/overcooked-v3-coot-response-candidates/omh17zwt \
  cilab-overcooked/overcooked-v3-coot-pipeline/9u2s6axd \
  cilab-overcooked/overcooked-v3-coot-response/ac9f9ssi \
  cilab-overcooked/overcooked-v3-coot-pipeline/yhnbq7jf \
  cilab-overcooked/overcooked-v3-coot-train/e22e0tlo \
  cilab-overcooked/overcooked-v3-coot-eval/agrn47k5
```
