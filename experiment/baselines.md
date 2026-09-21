# IPPO-CNN / IPPO-RNN / FCP sweep 실행 명령어

아래 순서대로 sweep을 생성하면 됩니다.

## IPPO-CNN

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-ippo_train experiment/ippo-cnn/train.yaml
wandb sweep --entity cilab-overcooked --project overcooked-v3-ippo_eval experiment/ippo-cnn/eval.yaml
```

## IPPO-RNN

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-ippo-rnn_train experiment/ippo-rnn/train.yaml
wandb sweep --entity cilab-overcooked --project overcooked-v3-ippo-rnn_eval experiment/ippo-rnn/eval.yaml
```

## FCP

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-fcp-population experiment/fcp/population.yaml
wandb sweep --entity cilab-overcooked --project overcooked-v3-fcp_train experiment/fcp/train.yaml
wandb sweep --entity cilab-overcooked --project overcooked-v3-fcp_eval experiment/fcp/eval.yaml
```

## Agent 실행 예시

Sweep 생성 후 출력되는 sweep id를 사용해서 agent를 실행합니다.

```bash
wandb agent cilab-overcooked/overcooked-v3-ippo_train/<SWEEP_ID>
```
