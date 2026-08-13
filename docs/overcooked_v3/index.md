# Overcooked V3

Overcooked V2에서 분리한 V3 동적 환경과 역할 형성 실험 문서다.

- [환경 개발 및 실험 workflow](workflow.md)
- [Hydra 및 W&B 학습 설정](training.md)
- [W&B artifact 기반 cross-play 평가](workflow.md#73-wb-run-artifact-기반-cross-play)
- [W&B project 기반 전체 cross-play matrix](workflow.md#74-wb-project-기반-전체-cross-play-matrix)
- [Cross-play 명령어 runbook](crossplay.md)
- [40-map cross-play sweep 실행](crossplay.md#41-기존-40개-맵-wb-sweep)
- [Role-scenario W&B sweep 실행 runbook](../../experiment/role_scenarios.md)

관련 실행기는 `scripts/overcooked_v3/`, Hydra 설정은 `conf/`, W&B sweep은
`experiment/sweeps/`에서 관리한다.
