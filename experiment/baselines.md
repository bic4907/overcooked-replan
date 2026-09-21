# IPPO-CNN / IPPO-RNN / FCP sweep 실행 명령어

아래 순서대로 sweep을 생성하면 됩니다.

## IPPO-CNN

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-ippo_train ippo-cnn/train.yaml
wandb sweep --entity cilab-overcooked --project overcooked-v3-ippo_eval ippo-cnn/eval.yaml
```

## IPPO-RNN

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-ippo-rnn_train ippo-rnn/train.yaml
wandb sweep --entity cilab-overcooked --project overcooked-v3-ippo-rnn_eval ippo-rnn/eval.yaml
```

## FCP

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-fcp-population fcp/population.yaml
wandb sweep --entity cilab-overcooked --project overcooked-v3-fcp_train fcp/train.yaml
wandb sweep --entity cilab-overcooked --project overcooked-v3-fcp_eval fcp/eval.yaml
```

## Agent 실행 예시

Sweep 생성 후 출력되는 sweep id를 사용해서 agent를 실행합니다.

```bash
wandb agent cilab-overcooked/overcooked-v3-ippo_train/<SWEEP_ID>
```
