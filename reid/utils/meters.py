from __future__ import absolute_import

import json
import os
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Optional, Iterable, Tuple


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ContinualResults:
    """
    Continual learning 실험에서 라운드(학습 단계)별 mAP / top-1 결과를
    데이터셋 단위로 저장하고, 나중에 불러와서 플롯/내보내기 할 수 있는 헬퍼.

    - 라운드는 1부터 시작하는 정수로 가정 (예: 1, 2, 3).
    - 특정 데이터셋이 일부 라운드에서만 평가됐다면, 나머지는 NaN으로 채워 플롯에서 선이 끊기게 처리.
    """

    def __init__(self, max_rounds: Optional[int] = None) -> None:
        # 내부 저장 구조: { dataset: { round: {"mAP": float, "top1": float} } }
        self._data: Dict[str, Dict[int, Dict[str, float]]] = {}
        self._max_rounds = max_rounds  # None이면 자동 계산

    # ---------------------------
    # 기록/조회
    # ---------------------------
    def add(self, dataset: str, round_idx: int, mAP: float, top1: float) -> None:
        """한 데이터셋의 특정 라운드 결과를 추가한다."""
        if round_idx < 1:
            raise ValueError("round_idx must start from 1.")
        self._data.setdefault(dataset, {})
        self._data[dataset][round_idx] = {"mAP": float(mAP), "top1": float(top1)}

    def max_rounds(self) -> int:
        """저장된 결과 기준으로 최대 라운드를 반환 (지정값이 있으면 그걸 사용)."""
        if self._max_rounds is not None:
            return self._max_rounds
        max_r = 0
        for per_ds in self._data.values():
            if per_ds:
                max_r = max(max_r, max(per_ds.keys()))
        return max_r if max_r > 0 else 1

    def datasets(self) -> Iterable[str]:
        return self._data.keys()

    def get_series(self, dataset: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        해당 데이터셋의 (mAP, top1) 시퀀스를 반환.
        길이는 max_rounds, 값이 없는 라운드는 np.nan.
        """
        R = self.max_rounds()
        mAPs = np.full(R, np.nan, dtype=float)
        top1s = np.full(R, np.nan, dtype=float)
        if dataset in self._data:
            for r, metrics in self._data[dataset].items():
                if 1 <= r <= R:
                    mAPs[r - 1] = metrics["mAP"]
                    top1s[r - 1] = metrics["top1"]
        return mAPs, top1s
    
    def get_average(self):
        mAP_list = []
        top1_list = []

        for ds in self.datasets():
            mAPs, top1s = self.get_series(ds)
            mAP_list.append(mAPs[-1])
            top1_list.append(top1s[-1])

        return np.mean(mAP_list), np.mean(top1_list)

    # ---------------------------
    # 저장/불러오기
    # ---------------------------
    def save_json(self, path: str) -> None:
        average_mAP, average_top1 = self.get_average()

        payload = {
            "max_rounds": self._max_rounds,
            "data": self._data,
            "average_mAP:": average_mAP,
            "average_top1": average_top1
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "ContinualResults":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        cr = cls(max_rounds=payload.get("max_rounds"))
        cr._data = {k: {int(r): v for r, v in d.items()} for k, d in payload["data"].items()}
        return cr

    # ---------------------------
    # 플로팅
    # ---------------------------
    def plot(
        self,
        title: str = "Continual Learning Results",
        metrics: Tuple[str, str] = ("mAP", "top1"),
        figsize: Tuple[int, int] = (10, 4),
        x_tick_labels: Optional[Iterable[str]] = None,
        legend_loc: str = "best",
        out_path: Optional[str] = None,
        tight_layout: bool = True,
        show: bool = True,
    ) -> None:
        """
        metrics=('mAP','top1')을 두 개의 서브플롯으로 그린다.
        각 데이터셋마다 선을 하나씩 그리며, 비어있는 라운드는 선이 끊긴다.
        """
        if metrics != ("mAP", "top1"):
            raise ValueError("Only ('mAP','top1') pair plotting is implemented for simplicity.")

        R = self.max_rounds()
        xs = np.arange(1, R + 1)
        if x_tick_labels is None:
            x_tick_labels = [f"T{r}" for r in xs]

        fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
        ax_map, ax_top1 = axes

        for ds in self.datasets():
            mAPs, top1s = self.get_series(ds)
            ax_map.plot(xs, mAPs, marker="o", label=ds)
            ax_top1.plot(xs, top1s, marker="o", label=ds)

        ax_map.set_title("mAP")
        ax_top1.set_title("Top-1")
        for ax in axes:
            ax.set_xlabel("Round")
            ax.set_xticks(xs)
            ax.set_xticklabels(list(x_tick_labels))
            ax.grid(True, linestyle="--", alpha=0.4)

        # 범례는 오른쪽 플롯에 표시
        ax_top1.legend(loc=legend_loc)
        fig.suptitle(title)

        if tight_layout:
            plt.tight_layout(rect=(0, 0, 1, 0.95))

        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            fig.savefig(out_path, dpi=200)

        if show:
            plt.show()
        plt.close(fig)

    # ---------------------------
    # 편의: 콘솔 보기 / CSV 내보내기
    # ---------------------------
    def to_table(self) -> str:
        """
        콘솔에 결과를 보기 좋게 출력하기 위한 문자열을 만든다.
        행: dataset, 열: round별 mAP / top1 쌍
        """
        R = self.max_rounds()
        header = ["dataset"] + [f"R{r} mAP" for r in range(1, R + 1)] + [f"R{r} top1" for r in range(1, R + 1)]
        rows = [header]
        for ds in self.datasets():
            mAPs, top1s = self.get_series(ds)
            row = [ds] + [f"{v:.3f}" if not np.isnan(v) else "-" for v in mAPs] + \
                        [f"{v:.3f}" if not np.isnan(v) else "-" for v in top1s]
            rows.append(row)

        # pretty print
        col_w = [max(len(str(rows[i][j])) for i in range(len(rows))) for j in range(len(rows[0]))]
        lines = []
        for r, row in enumerate(rows):
            line = " | ".join(str(cell).ljust(col_w[j]) for j, cell in enumerate(row))
            if r == 0:
                sep = "-+-".join("-" * col_w[j] for j in range(len(row)))
                lines.append(line)
                lines.append(sep)
            else:
                lines.append(line)
        return "\n".join(lines)

    def export_csv(self, path: str) -> None:
        """
        CSV 내보내기 (wide format). 열 = dataset, R1_mAP, R1_top1, R2_mAP, ...
        """
        R = self.max_rounds()
        header = ["dataset"] + [f"R{r}_mAP" for r in range(1, R + 1)] + [f"R{r}_top1" for r in range(1, R + 1)]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for ds in self.datasets():
                mAPs, top1s = self.get_series(ds)
                mAP_str = [f"{v:.6f}" if not np.isnan(v) else "" for v in mAPs]
                top1_str = [f"{v:.6f}" if not np.isnan(v) else "" for v in top1s]
                f.write(",".join([ds] + mAP_str + top1_str) + "\n")
