"""
HouseDiffusion 결과 시각화 스크립트.

outputs/set8/sample_*.npz 파일들을 읽어서 평면도 PNG로 저장합니다.

사용법:
    python scripts/visualize_samples.py
    python scripts/visualize_samples.py --target_set 8 --max_samples 10

발표용 이미지가 outputs/set8/png/ 에 저장됩니다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # GUI 없이 파일로만 저장
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection


# ---------------------------------------------------------------------------
# RPLAN 룸 색상 / 라벨 (논문 Fig.4와 동일한 팔레트)
# ---------------------------------------------------------------------------
ROOM_COLORS = {
    1:  "#EE4D4D",  # Living Room
    2:  "#C67C7B",  # Kitchen
    3:  "#FFD274",  # Bedroom
    4:  "#BEBEBE",  # Bathroom
    5:  "#BFE3E8",  # Balcony
    6:  "#7BA779",  # Entrance
    7:  "#E87A90",  # Dining Room
    8:  "#FF8C69",  # Study Room
    10: "#1F849B",  # Storage
    11: "#727171",  # Front Door
    12: "#D3A2C7",  # Interior Door
    13: "#785A67",  # Unknown
}

ROOM_LABELS = {
    1: "Living", 2: "Kitchen", 3: "Bedroom", 4: "Bathroom",
    5: "Balcony", 6: "Entrance", 7: "Dining", 8: "Study",
    10: "Storage", 11: "FrontDoor", 12: "IntDoor", 13: "Unknown",
}


def load_sample(npz_path: Path) -> dict:
    """npz 파일 로드 → dict로 반환."""
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def parse_one_house(sample: np.ndarray, room_types: np.ndarray,
                    room_indices: np.ndarray, padding_mask: np.ndarray = None):
    """
    한 평면도(sample 1개)에서 방별 폴리곤을 뽑아낸다.

    sample: (T, 2) 또는 (2, T) 좌표
    room_types: (T, 25) one-hot
    room_indices: (T, 32) one-hot
    padding_mask: (T,) — 1=실제 코너, 0=패딩
    """
    # sample이 (2, T) 면 transpose
    if sample.shape[0] == 2 and sample.shape[1] != 2:
        sample = sample.T  # (T, 2)

    T = sample.shape[0]

    # 패딩 제거 (있으면)
    if padding_mask is not None:
        mask = padding_mask.astype(bool)
        if mask.shape[0] == T:
            sample = sample[mask]
            room_types = room_types[mask]
            room_indices = room_indices[mask]

    # 방별로 그룹화
    rt_idx = np.argmax(room_types, axis=1) if room_types.ndim == 2 else room_types
    ri_idx = np.argmax(room_indices, axis=1) if room_indices.ndim == 2 else room_indices

    rooms = {}
    for coord, rtype, ridx in zip(sample, rt_idx, ri_idx):
        if ridx not in rooms:
            rooms[ridx] = {"type": int(rtype), "corners": []}
        rooms[ridx]["corners"].append(coord)

    # 좌표가 [-1, 1] 범위 → [0, 256]로
    out = []
    for ridx, info in rooms.items():
        corners = np.array(info["corners"])
        if len(corners) < 3:
            continue
        corners = (corners + 1) / 2 * 256
        out.append({
            "type": info["type"],
            "index": int(ridx),
            "corners": corners,
        })
    return out


def plot_house(rooms, ax, title=""):
    """방 폴리곤들을 ax에 그린다."""
    patches = []
    colors = []
    for room in rooms:
        rtype = room["type"]
        color = ROOM_COLORS.get(rtype, "#CCCCCC")
        poly = MplPolygon(room["corners"], closed=True)
        patches.append(poly)
        colors.append(color)

    p = PatchCollection(patches, alpha=0.85, edgecolor="black", linewidths=1.5)
    p.set_facecolor(colors)
    ax.add_collection(p)

    # 라벨
    for room in rooms:
        cx, cy = room["corners"].mean(axis=0)
        label = ROOM_LABELS.get(room["type"], f"T{room['type']}")
        ax.text(cx, cy, label, fontsize=7, ha="center", va="center")

    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.invert_yaxis()  # 이미지 좌표계 (y 아래로)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)


def visualize_npz(npz_path: Path, out_dir: Path, max_houses: int = 4):
    """npz 1개 → 그 안의 평면도들을 PNG로 저장."""
    data = load_sample(npz_path)

    sample = data["sample"]  # (B, 2, T) 또는 (B, T, 2)

    # batch 차원 정리
    if sample.ndim == 4:
        sample = sample[-1]  # 마지막 timestep (denoising 끝난 결과)
    if sample.ndim == 3:
        # (B, 2, T) 또는 (B, T, 2)
        B = sample.shape[0]
    else:
        B = 1
        sample = sample[None]

    # 조건 데이터
    room_types = data.get("room_types")
    room_indices = data.get("room_indices")
    padding_mask = data.get("src_key_padding_mask")

    # 보통 batch_size=1로 돌렸으니 B=1 인데,
    # 안에 여러 평면도가 한 번에 들어있을 수도 있음
    n_to_plot = min(B, max_houses)

    for b in range(n_to_plot):
        s = sample[b]
        rt = room_types[b] if room_types is not None and room_types.ndim >= 2 else room_types
        ri = room_indices[b] if room_indices is not None and room_indices.ndim >= 2 else room_indices
        pm = None
        if padding_mask is not None:
            pm = padding_mask[b] if padding_mask.ndim >= 2 else padding_mask
            # 1 - mask 형태일 수 있음 (rplanhg_datasets에서 1-padding_mask로 줌)
            # 실제 코너는 padding_mask=1인 곳

        try:
            rooms = parse_one_house(s, rt, ri, padding_mask=pm)
        except Exception as e:
            print(f"  [skip] {npz_path.name}[{b}] 파싱 실패: {e}")
            continue

        if not rooms:
            print(f"  [skip] {npz_path.name}[{b}] 방 없음")
            continue

        fig, ax = plt.subplots(figsize=(5, 5))
        plot_house(rooms, ax, title=f"{npz_path.stem} (house {b})")
        out_path = out_dir / f"{npz_path.stem}_h{b:02d}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_set", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=20,
                        help="처리할 npz 파일 최대 개수")
    parser.add_argument("--max_houses_per_sample", type=int, default=4,
                        help="한 npz 안에서 최대 몇 개의 평면도를 그릴지")
    args = parser.parse_args()

    in_dir = Path("outputs") / f"set{args.target_set}"
    out_dir = in_dir / "png"
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(in_dir.glob("sample_*.npz"))[: args.max_samples]
    if not npz_files:
        print(f"npz 파일이 없습니다: {in_dir}")
        return

    print(f"{len(npz_files)}개 npz 파일 처리 중...")
    for npz in npz_files:
        print(f"\n[{npz.name}]")
        visualize_npz(npz, out_dir, max_houses=args.max_houses_per_sample)

    print(f"\n완료! PNG 이미지: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
