# -*- coding: utf-8 -*-
"""
MapleStory "한글 모아모아" helper/bot.

Windows / Python 3.10+.
The program uses screen recognition + a beam-search block-placement solver,
then controls the mouse with PyAutoGUI.

IMPORTANT:
- Start the mini-game yourself and keep the game visible.
- For the most reliable skill tracking, start a fresh round (0 skill charges)
  before starting this program.
- F9 = emergency stop
- F8 = pause/resume
- F6 = recalibrate
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import threading
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import cv2
import numpy as np
import pyautogui

try:
    import keyboard
except Exception:
    keyboard = None

# Windows DPI awareness: prevents Windows display scaling from making
# PyAutoGUI coordinates disagree with screenshot coordinates.
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

CONFIG_PATH = Path(__file__).with_name("hangul_moa_config.json")

COLS = 10
ROWS = 16
FULL_MASK = (1 << COLS) - 1

STOP = threading.Event()
PAUSE = threading.Event()
PAUSE.clear()


@dataclass
class Piece:
    slot: int
    cells: tuple[tuple[int, int], ...]   # normalized (r,c)
    anchor: tuple[int, int]              # block used as mouse-drag anchor
    source_xy: tuple[int, int]            # screen coordinate of anchor block

    @property
    def size(self):
        return len(self.cells)


def norm_shape(cells):
    cells = list(cells)
    min_r = min(r for r, c in cells)
    min_c = min(c for r, c in cells)
    out = sorted((r - min_r, c - min_c) for r, c in cells)
    return tuple(out)


def rotate_shape(cells):
    # 90 degrees clockwise in grid coordinates.
    return norm_shape([(c, -r) for r, c in cells])


def flip_shape(cells):
    # Horizontal mirror.
    return norm_shape([(r, -c) for r, c in cells])


def all_orientations(cells):
    seen = set()
    cur = norm_shape(cells)
    for _ in range(4):
        for s in (cur, flip_shape(cur)):
            s = norm_shape(s)
            if s not in seen:
                seen.add(s)
        cur = rotate_shape(cur)
    return list(seen)


def transform_by_sequence(cells, seq):
    s = norm_shape(cells)
    for a in seq:
        s = rotate_shape(s) if a == "R" else flip_shape(s)
    return s


def find_transform_sequence(src, dst):
    src = norm_shape(src)
    dst = norm_shape(dst)
    if src == dst:
        return []
    # The dihedral group is tiny; BFS is enough.
    q = [(src, [])]
    seen = {src}
    while q:
        s, path = q.pop(0)
        for a in ("R", "F"):
            ns = rotate_shape(s) if a == "R" else flip_shape(s)
            if ns in seen:
                continue
            npth = path + [a]
            if ns == dst:
                return npth
            seen.add(ns)
            if len(npth) < 8:
                q.append((ns, npth))
    return []


def shape_masks(cells):
    """Convert a normalized shape into row bit masks."""
    d = {}
    for r, c in cells:
        d[r] = d.get(r, 0) | (1 << c)
    return tuple(sorted(d.items()))


def place_rows(rows_bits, shape, top, left):
    """Return (new_rows, cleared_count, valid)."""
    rows2 = list(rows_bits)
    for r, c in shape:
        rr = top + r
        cc = left + c
        if rr < 0 or rr >= ROWS or cc < 0 or cc >= COLS:
            return None, 0, False
        if rows2[rr] & (1 << cc):
            return None, 0, False
    for r, c in shape:
        rows2[top + r] |= 1 << (left + c)
    cleared = 0
    for r in range(ROWS):
        if rows2[r] == FULL_MASK:
            rows2[r] = 0
            cleared += 1
    return tuple(rows2), cleared, True


def legal_moves(rows_bits, shape):
    h = max(r for r, c in shape) + 1
    w = max(c for r, c in shape) + 1
    for r in range(ROWS - h + 1):
        for c in range(COLS - w + 1):
            nr, cleared, ok = place_rows(rows_bits, shape, r, c)
            if ok:
                yield r, c, nr, cleared


def row_quality(rows_bits):
    counts = [int(x.bit_count()) for x in rows_bits]
    # Completing/near-completing rows is valuable because it creates
    # future line clears without forcing a particular column height.
    near8 = sum(1 for x in counts if x == 8)
    near7 = sum(1 for x in counts if x == 7)
    near6 = sum(1 for x in counts if x == 6)

    # Penalize very dense boards slightly. This prevents the solver from
    # greedily filling cells without preserving maneuvering room.
    occupied = sum(counts)

    # Penalize abrupt row-density changes. It is only a weak tie-breaker.
    rough = sum(abs(counts[i] - counts[i + 1]) for i in range(ROWS - 1))
    return 22 * near8 + 8 * near7 + 2 * near6 - 0.25 * occupied - 0.25 * rough


def mobility(rows_bits, shapes):
    total = 0
    for s in shapes:
        total += sum(1 for _ in legal_moves(rows_bits, s))
    return min(total, 80)


def evaluate_state(rows_bits, score, remaining_shapes):
    return score + row_quality(rows_bits) + 0.35 * mobility(rows_bits, remaining_shapes)


def solve_current_pieces(rows_bits, pieces, beam_width=45):
    """
    Beam-search all useful orderings of the currently available pieces.
    Returns the first move (slot, orientation, row, col) from the best
    sequence found.
    """
    if not pieces:
        return None

    # State: (board, used_mask, score, first_move, depth)
    states = [(tuple(rows_bits), 0, 0, None, 0)]

    # We can use each currently available piece at most once before new
    # pieces appear. Search depth up to all available pieces.
    depth_limit = len(pieces)

    for depth in range(depth_limit):
        candidates = []
        for board, used, score, first, _ in states:
            for i, p in enumerate(pieces):
                if used & (1 << i):
                    continue
                for orient in all_orientations(p.cells):
                    for r, c, nb, cleared in legal_moves(board, orient):
                        gain = len(orient) + 300 * (cleared ** 2)
                        ns = score + gain
                        fm = first
                        if fm is None:
                            fm = (p.slot, orient, r, c)
                        candidates.append(
                            (
                                evaluate_state(
                                    nb,
                                    ns,
                                    [
                                        q.cells
                                        for j, q in enumerate(pieces)
                                        if not (used | (1 << i)) & (1 << j)
                                    ],
                                ),
                                nb,
                                used | (1 << i),
                                ns,
                                fm,
                                depth + 1,
                            )
                        )
        if not candidates:
            break
        candidates.sort(key=lambda x: x[0], reverse=True)
        states = [
            (x[1], x[2], x[3], x[4], x[5])
            for x in candidates[:beam_width]
        ]

    if not states:
        return None
    best = max(states, key=lambda x: evaluate_state(x[0], x[3], []))
    return best[3], best[2], best[3], best[4]


def load_config():
    if not CONFIG_PATH.exists():
        return None
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_config(cfg):
    CONFIG_PATH.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def calibrate():
    print("\n=== 한글 모아모아 봇 보정 ===")
    print("게임 창을 화면에 띄워두세요.")
    print("보드의 '왼쪽 위 칸 정중앙'에 마우스를 올린 뒤 Enter.")
    input()
    p1 = pyautogui.position()

    print("이번에는 보드의 '오른쪽 아래 칸 정중앙'에 마우스를 올린 뒤 Enter.")
    input()
    p2 = pyautogui.position()

    cell_x = (p2.x - p1.x) / (COLS - 1)
    cell_y = (p2.y - p1.y) / (ROWS - 1)

    if cell_x < 10 or cell_y < 10:
        raise RuntimeError("보정값이 이상합니다. 게임 보드 안쪽의 칸 중앙을 다시 지정하세요.")

    cfg = {
        "board_x0": int(p1.x),
        "board_y0": int(p1.y),
        "cell_x": float(cell_x),
        "cell_y": float(cell_y),
    }
    save_config(cfg)
    print(f"저장 완료: {CONFIG_PATH}")
    print(cfg)
    return cfg


def board_center(cfg, r, c):
    return (
        round(cfg["board_x0"] + c * cfg["cell_x"]),
        round(cfg["board_y0"] + r * cfg["cell_y"]),
    )


def panel_geometry(cfg):
    """
    Relative layout derived from the supplied game screenshots.
    The x/y scaling is taken from the user's own board calibration, so
    Windows display scaling does not matter.
    """
    bx, by = cfg["board_x0"], cfg["board_y0"]
    cx, cy = cfg["cell_x"], cfg["cell_y"]

    # The piece cards sit to the right of the 10x16 board.
    panel_x1 = round(bx + 10.15 * cx)
    panel_x2 = round(bx + 12.7 * cx)

    # Three card bands. These are intentionally generous because a piece
    # can be several cells tall.
    bands = [
        (by + 0.35 * cy, by + 3.65 * cy),
        (by + 3.65 * cy, by + 6.55 * cy),
        (by + 6.55 * cy, by + 9.55 * cy),
    ]

    # Rotate / flip buttons are to the right of the piece icon.
    button_x = round(bx + 13.05 * cx)
    rotate_ys = [
        round(by + 1.75 * cy),
        round(by + 4.85 * cy),
        round(by + 7.75 * cy),
    ]
    flip_ys = [round(y + 0.92 * cy) for y in rotate_ys]

    dot_button = (round(bx + 12.55 * cx), round(by + 14.15 * cy))
    swap_button = (round(bx + 12.55 * cx), round(by + 15.45 * cy))

    return {
        "panel_x1": panel_x1,
        "panel_x2": panel_x2,
        "bands": bands,
        "button_x": button_x,
        "rotate_ys": rotate_ys,
        "flip_ys": flip_ys,
        "dot_button": dot_button,
        "swap_button": swap_button,
    }


def screenshot_bgr():
    img = pyautogui.screenshot()
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def detect_board(img, cfg):
    bx, by = cfg["board_x0"], cfg["board_y0"]
    cx, cy = cfg["cell_x"], cfg["cell_y"]

    out = []
    # Move the mouse away before calling this, so the cursor does not look
    # like an occupied cell.
    for r in range(ROWS):
        row = 0
        for c in range(COLS):
            x = round(bx + c * cx)
            y = round(by + r * cy)
            hw = max(5, int(cx * 0.25))
            hh = max(5, int(cy * 0.25))
            patch = img[
                max(0, y - hh):y + hh + 1,
                max(0, x - hw):x + hw + 1,
            ]
            if patch.size == 0:
                continue

            # Empty cells are almost uniform in their central area.
            # Real blocks have strong color/highlight variation.
            p = patch.astype(np.float32)
            std = float(p.reshape(-1, 3).std(axis=0).mean())
            mean = p.reshape(-1, 3).mean(axis=0)

            # A second check rejects tiny skill icons sitting in otherwise
            # empty cells.
            center_var = float(p.reshape(-1, 3).std(axis=0).max())

            occupied = std > 5.0 and center_var > 3.5
            if occupied:
                row |= 1 << c
        out.append(row)
    return tuple(out)


def detect_piece_in_band(img, cfg, band):
    g = panel_geometry(cfg)
    x1, x2 = g["panel_x1"], g["panel_x2"]
    y1, y2 = map(int, band)

    x1 = max(0, int(x1))
    x2 = min(img.shape[1], int(x2))
    y1 = max(0, y1)
    y2 = min(img.shape[0], y2)

    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Block fills are strongly saturated. The white card and text are not.
    mask = cv2.inRange(
        hsv,
        np.array([0, 115, 80], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )

    # Remove tiny noise.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cx, cy = cfg["cell_x"], cfg["cell_y"]
    comps = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if (
            0.45 * cx <= w <= 1.25 * cx
            and 0.45 * cy <= h <= 1.25 * cy
            and 0.20 * cx * cy <= area <= 1.2 * cx * cy
        ):
            sx = x1 + x + w / 2
            sy = y1 + y + h / 2
            comps.append((sx, sy))

    if not comps:
        return None

    # Normalize square centers to a local lattice.
    minx = min(x for x, y in comps)
    miny = min(y for x, y in comps)
    local = []
    for x, y in comps:
        cc = int(round((x - minx) / cx))
        rr = int(round((y - miny) / cy))
        local.append((rr, cc, x, y))

    cells = norm_shape([(r, c) for r, c, _, _ in local])
    # Pick a real component as the drag anchor. Keep its normalized coord.
    anchor_raw = local[0]
    anchor_norm = (
        int(anchor_raw[0] - min(r for r, c in [(z[0], z[1]) for z in local])),
        int(anchor_raw[1] - min(c for r, c in [(z[0], z[1]) for z in local])),
    )

    # Better anchor calculation using the normalized set.
    raw_norms = [
        (
            int(round((x - minx) / cx)),
            int(round((y - miny) / cy)),
            x,
            y,
        )
        for x, y in comps
    ]
    raw_min_r = min(z[0] for z in raw_norms)
    raw_min_c = min(z[1] for z in raw_norms)
    raw_norms = [(r - raw_min_r, c - raw_min_c, x, y) for r, c, x, y in raw_norms]

    ar, ac, ax, ay = raw_norms[0]
    return {
        "cells": cells,
        "anchor": (ar, ac),
        "source_xy": (int(ax), int(ay)),
        "center_xy": (
            int(np.mean([z[2] for z in raw_norms])),
            int(np.mean([z[3] for z in raw_norms])),
        ),
    }


def detect_pieces(img, cfg):
    g = panel_geometry(cfg)
    pieces = []
    for slot, band in enumerate(g["bands"]):
        d = detect_piece_in_band(img, cfg, band)
        if d:
            pieces.append(
                Piece(
                    slot=slot,
                    cells=tuple(d["cells"]),
                    anchor=tuple(d["anchor"]),
                    source_xy=tuple(d["source_xy"]),
                )
            )
    return pieces


def wait_board_stable(cfg, timeout=1.2):
    pyautogui.moveTo(
        round(cfg["board_x0"] + 12.0 * cfg["cell_x"]),
        round(cfg["board_y0"] + 10.0 * cfg["cell_y"]),
        duration=0,
    )
    last = None
    stable = 0
    deadline = time.time() + timeout
    while time.time() < deadline and not STOP.is_set():
        img = screenshot_bgr()
        b = detect_board(img, cfg)
        if b == last:
            stable += 1
            if stable >= 2:
                return b
        else:
            stable = 0
            last = b
        time.sleep(0.08)
    return last


def click_rotate_or_flip(cfg, slot, current, target):
    g = panel_geometry(cfg)
    seq = find_transform_sequence(current, target)

    for action in seq:
        if STOP.is_set():
            return False
        if action == "R":
            xy = (g["button_x"], g["rotate_ys"][slot])
        else:
            xy = (g["button_x"], g["flip_ys"][slot])
        pyautogui.click(*xy)
        time.sleep(0.10)
    return True


def drag_piece(cfg, piece, target_shape, top, left):
    # Transform the piece before dragging.
    if not click_rotate_or_flip(cfg, piece.slot, piece.cells, target_shape):
        return False

    # We need to know where the chosen anchor block ended up after
    # transformation. Recompute it by transforming the anchor coordinate.
    s = piece.cells
    a = piece.anchor
    # Find where that specific anchor block maps under the same action sequence.
    seq = find_transform_sequence(piece.cells, target_shape)
    ar, ac = a
    for action in seq:
        if action == "R":
            ar, ac = ac, -ar
        else:
            ar, ac = ar, -ac

    # Normalize the transformed anchor together with the target shape.
    allr = [r for r, c in target_shape]
    allc = [c for r, c in target_shape]
    minr, minc = min(allr), min(allc)
    ar -= minr
    ac -= minc

    # Source coordinate is the original block we clicked. The actual
    # panel piece has rotated in place, so after transformation the chosen
    # anchor is still at the corresponding relative cell.
    tx, ty = board_center(cfg, top + ar, left + ac)

    pyautogui.moveTo(*piece.source_xy, duration=0.05)
    pyautogui.mouseDown()
    pyautogui.moveTo(tx, ty, duration=0.18)
    pyautogui.mouseUp()
    time.sleep(0.16)
    return True


def button_region(cfg, which):
    g = panel_geometry(cfg)
    x, y = g["dot_button"] if which == "dot" else g["swap_button"]
    cx, cy = cfg["cell_x"], cfg["cell_y"]
    # The count digit sits toward the right side of the pill button.
    return (
        int(x - 0.9 * cx),
        int(y - 0.45 * cy),
        int(x + 1.0 * cx),
        int(y + 0.45 * cy),
    )


def crop_region(img, box):
    x1, y1, x2, y2 = box
    return img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]


def skill_baseline(img, cfg):
    return {
        "dot": crop_region(img, button_region(cfg, "dot")).copy(),
        "swap": crop_region(img, button_region(cfg, "swap")).copy(),
    }


def skill_changed(img, cfg, baseline, which):
    ref = baseline.get(which)
    cur = crop_region(img, button_region(cfg, which))
    if ref is None or cur.size == 0 or ref.size == 0 or cur.shape != ref.shape:
        return False
    a = cv2.absdiff(cur, ref).astype(np.float32)
    # The background stays essentially identical; the count glyph changes.
    return float(a.mean()) > 2.0


def use_dot(cfg, target):
    g = panel_geometry(cfg)
    pyautogui.click(*g["dot_button"])
    time.sleep(0.08)
    pyautogui.click(*board_center(cfg, target[0], target[1]))
    time.sleep(0.25)


def use_swap(cfg, piece):
    g = panel_geometry(cfg)
    pyautogui.click(*g["swap_button"])
    time.sleep(0.08)
    pyautogui.click(*piece.source_xy)
    time.sleep(0.35)


def find_one_gap_row(board):
    for r, bits in enumerate(board):
        if bits.bit_count() == COLS - 1:
            for c in range(COLS):
                if not (bits & (1 << c)):
                    return r, c
    return None


def has_clear_move(board, pieces):
    for p in pieces:
        for o in all_orientations(p.cells):
            for _, _, _, cleared in legal_moves(board, o):
                if cleared:
                    return True
    return False


def print_debug(cfg):
    pyautogui.moveTo(
        round(cfg["board_x0"] + 12.0 * cfg["cell_x"]),
        round(cfg["board_y0"] + 11.0 * cfg["cell_y"]),
        duration=0,
    )
    time.sleep(0.15)
    img = screenshot_bgr()
    board = detect_board(img, cfg)
    pieces = detect_pieces(img, cfg)
    print("\n--- DEBUG ---")
    print("board rows (top -> bottom):")
    for row in board:
        print(format(row, f"0{COLS}b")[::-1])
    print("pieces:")
    for p in pieces:
        print(f" slot={p.slot+1} cells={p.cells} anchor={p.anchor} src={p.source_xy}")
    print("------------\n")


def setup_hotkeys():
    if keyboard is None:
        print("주의: keyboard 모듈을 불러오지 못했습니다. F8/F9 전역키는 사용할 수 없습니다.")
        return
    try:
        keyboard.add_hotkey("f9", STOP.set)
        keyboard.add_hotkey("f8", lambda: (PAUSE.set() if not PAUSE.is_set() else PAUSE.clear()))
        keyboard.add_hotkey("f6", lambda: calibrate())
    except Exception as e:
        print("전역 단축키 설정 실패:", e)


def wait_if_paused():
    while PAUSE.is_set() and not STOP.is_set():
        time.sleep(0.15)


def run_bot(cfg, dry_run=False):
    setup_hotkeys()

    print("\n5초 후 시작합니다.")
    print("F8 = 일시정지/재개, F9 = 즉시 정지, F6 = 재보정")
    for n in range(5, 0, -1):
        print(n)
        time.sleep(1)

    # Start with cursor away from the board.
    pyautogui.moveTo(
        round(cfg["board_x0"] + 12.5 * cfg["cell_x"]),
        round(cfg["board_y0"] + 12.0 * cfg["cell_y"]),
        duration=0,
    )
    img0 = screenshot_bgr()
    baseline = skill_baseline(img0, cfg)

    placement_count = 0

    while not STOP.is_set():
        wait_if_paused()
        if STOP.is_set():
            break

        pyautogui.moveTo(
            round(cfg["board_x0"] + 12.0 * cfg["cell_x"]),
            round(cfg["board_y0"] + 11.0 * cfg["cell_y"]),
            duration=0,
        )
        img = screenshot_bgr()
        board = detect_board(img, cfg)
        pieces = detect_pieces(img, cfg)

        if not pieces:
            # Either the round ended or the UI is mid-animation.
            time.sleep(0.35)
            continue

        # Strategic use of "점 찍기": if no currently-held piece can clear
        # a row but a row has exactly one empty cell, use the skill to clear it.
        if not dry_run and skill_changed(img, cfg, baseline, "dot"):
            gap = find_one_gap_row(board)
            if gap and not has_clear_move(board, pieces):
                print("점 찍기:", gap)
                use_dot(cfg, gap)
                wait_board_stable(cfg)
                continue

        # If no piece can be placed at all, use "바꿔 뽑기" if available.
        any_move = False
        for p in pieces:
            if any(legal_moves(board, o) for o in all_orientations(p.cells)):
                any_move = True
                break

        if not any_move:
            if not dry_run and skill_changed(img, cfg, baseline, "swap"):
                # Replace the largest piece first; it is the hardest one to
                # place when the board is tight.
                p = max(pieces, key=lambda z: z.size)
                print("바꿔 뽑기: slot", p.slot + 1)
                use_swap(cfg, p)
                continue

            print("더 이상 놓을 수 있는 조각이 없습니다. 종료.")
            break

        # Solve the current set.
        result = solve_current_pieces(board, pieces)
        if result is None:
            print("해결할 수 있는 수를 찾지 못했습니다.")
            break

        _, _, _, first = result
        if first is None:
            break

        slot, target_shape, top, left = first
        piece = next((p for p in pieces if p.slot == slot), None)
        if piece is None:
            continue

        print(
            f"#{placement_count+1}: slot={slot+1}, "
            f"shape={target_shape}, pos=({top},{left}), cells={len(target_shape)}"
        )

        if dry_run:
            placement_count += 1
            # Do not actually change the board in dry-run.
            time.sleep(0.15)
            break

        ok = drag_piece(cfg, piece, target_shape, top, left)
        if not ok:
            break

        placement_count += 1
        wait_board_stable(cfg)

    print("봇 종료. 총 배치 시도:", placement_count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true", help="보드 좌표 재보정")
    parser.add_argument("--debug", action="store_true", help="현재 화면에서 보드/조각 인식만 출력")
    parser.add_argument("--dry-run", action="store_true", help="마우스를 움직이지 않고 첫 수만 계산")
    args = parser.parse_args()

    cfg = load_config()
    if args.calibrate or cfg is None:
        cfg = calibrate()

    if args.debug:
        print_debug(cfg)
        return

    run_bot(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
