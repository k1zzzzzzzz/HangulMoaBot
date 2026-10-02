# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyautogui

try:
    import keyboard
except Exception:
    keyboard = None


# ============================================================
# 기본 설정
# ============================================================

COLS = 10
ROWS = 16
FULL_MASK = (1 << COLS) - 1

CONFIG_FILE = Path("hangul_moa_bot_config.json")
DEBUG_DIR = Path("debug")

pyautogui.PAUSE = 0.04
pyautogui.FAILSAFE = True


# ============================================================
# 데이터
# ============================================================

@dataclass
class Piece:
    slot: int
    cells: tuple
    source_xy: tuple
    bbox: tuple
    confidence: float


# ============================================================
# DPI
# ============================================================

def set_dpi_awareness():
    if os.name != "nt":
        return

    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


# ============================================================
# 화면
# ============================================================

def screenshot():
    return np.array(pyautogui.screenshot())


def save_debug(name, img):
    DEBUG_DIR.mkdir(exist_ok=True)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(DEBUG_DIR / name), img)


# ============================================================
# 도형
# ============================================================

def normalize_shape(cells):
    if not cells:
        return tuple()

    min_r = min(r for r, c in cells)
    min_c = min(c for r, c in cells)

    return tuple(
        sorted(
            (r - min_r, c - min_c)
            for r, c in cells
        )
    )


def rotate_shape(shape):
    return normalize_shape(
        [(c, -r) for r, c in shape]
    )


def flip_shape(shape):
    return normalize_shape(
        [(r, -c) for r, c in shape]
    )


def orientations(shape):
    result = []
    seen = set()

    original = normalize_shape(shape)

    for flipped in range(2):

        cur = flip_shape(original) if flipped else original

        for _ in range(4):

            cur = normalize_shape(cur)

            if cur not in seen:
                seen.add(cur)
                result.append(cur)

            cur = rotate_shape(cur)

    return result


def transform_sequence(original, target):
    target = normalize_shape(target)

    base = normalize_shape(original)

    for flipped in range(2):

        cur = flip_shape(base) if flipped else base
        sequence = ["F"] if flipped else []

        for _ in range(4):

            if normalize_shape(cur) == target:
                return sequence

            cur = rotate_shape(cur)
            sequence.append("R")

    return None


# ============================================================
# 보드
# ============================================================

def place_piece(board, shape, top, left):

    rows = list(board)

    for dr, dc in shape:

        r = top + dr
        c = left + dc

        if r < 0 or r >= ROWS:
            return None

        if c < 0 or c >= COLS:
            return None

        bit = 1 << c

        if rows[r] & bit:
            return None

        rows[r] |= bit

    cleared = 0

    for r in range(ROWS):

        if rows[r] == FULL_MASK:
            rows[r] = 0
            cleared += 1

    return tuple(rows), cleared


def legal_moves(board, shape):

    if not shape:
        return

    h = max(r for r, c in shape) + 1
    w = max(c for r, c in shape) + 1

    for r in range(ROWS - h + 1):

        for c in range(COLS - w + 1):

            result = place_piece(
                board,
                shape,
                r,
                c
            )

            if result is not None:

                new_board, cleared = result

                yield r, c, new_board, cleared


# ============================================================
# 보드 평가
# ============================================================

def board_score(board):

    counts = [
        row.bit_count()
        for row in board
    ]

    near7 = sum(x == 7 for x in counts)
    near8 = sum(x == 8 for x in counts)
    near9 = sum(x == 9 for x in counts)

    roughness = sum(
        abs(counts[i] - counts[i + 1])
        for i in range(ROWS - 1)
    )

    occupied = sum(counts)

    score = 0

    score += near7 * 8
    score += near8 * 20
    score += near9 * 35

    score -= roughness * 0.25
    score -= occupied * 0.05

    return score


def mobility(board, pieces):

    total = 0

    for p in pieces:

        for shape in orientations(p.cells):

            total += sum(
                1
                for _ in legal_moves(board, shape)
            )

    return min(total, 100)


# ============================================================
# AI
# ============================================================

def solve(board, pieces, beam_width=60):

    if not pieces:
        return None

    # board, 사용한 조각 비트, 점수, 첫 수
    states = [
        (tuple(board), 0, 0.0, None)
    ]

    for depth in range(len(pieces)):

        candidates = []

        for state_board, used, score, first in states:

            for i, piece in enumerate(pieces):

                if used & (1 << i):
                    continue

                for shape in orientations(piece.cells):

                    for r, c, new_board, cleared in legal_moves(
                        state_board,
                        shape
                    ):

                        gain = len(shape)

                        if cleared == 1:
                            gain += 250
                        elif cleared == 2:
                            gain += 700
                        elif cleared == 3:
                            gain += 1400
                        elif cleared >= 4:
                            gain += 2500

                        new_score = score + gain

                        if first is None:
                            first_move = (
                                piece.slot,
                                shape,
                                r,
                                c
                            )
                        else:
                            first_move = first

                        remaining = [
                            p
                            for j, p in enumerate(pieces)
                            if not ((used | (1 << i)) & (1 << j))
                        ]

                        evaluation = (
                            new_score
                            + board_score(new_board)
                            + mobility(
                                new_board,
                                remaining
                            ) * 0.30
                        )

                        candidates.append(
                            (
                                evaluation,
                                new_board,
                                used | (1 << i),
                                new_score,
                                first_move
                            )
                        )

        if not candidates:
            break

        candidates.sort(
            key=lambda x: x[0],
            reverse=True
        )

        # 같은 보드 상태 제거
        unique = []
        seen = set()

        for candidate in candidates:

            key = (
                candidate[1],
                candidate[2]
            )

            if key in seen:
                continue

            seen.add(key)
            unique.append(candidate)

            if len(unique) >= beam_width:
                break

        states = [
            (
                x[1],
                x[2],
                x[3],
                x[4]
            )
            for x in unique
        ]

    if not states:
        return None

    best = max(
        states,
        key=lambda x: (
            x[2] +
            board_score(x[0])
        )
    )

    return best[3]


# ============================================================
# 설정
# ============================================================

def default_config():

    return {
        "board_x": 500,
        "board_y": 200,

        "cell_x": 32,
        "cell_y": 32,

        "panel_x0": 850,
        "panel_y0": 200,

        "panel_x1": 1100,
        "panel_y1": 650,

        "rotate_x": 1080,

        "rotate_y": [
            260,
            380,
            500
        ],

        "flip_y": [
            290,
            410,
            530
        ],

        "dot_button": [
            1000,
            680
        ],

        "swap_button": [
            1000,
            725
        ]
    }


def load_config():

    if not CONFIG_FILE.exists():
        return default_config()

    try:

        data = json.loads(
            CONFIG_FILE.read_text(
                encoding="utf-8"
            )
        )

        config = default_config()
        config.update(data)

        return config

    except Exception:

        return default_config()


def save_config(config):

    CONFIG_FILE.write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


# ============================================================
# 보정
# ============================================================

def calibrate(config):

    print()
    print("=" * 60)
    print("보정 시작")
    print("=" * 60)

    print()
    print("① 보드 왼쪽 위 칸의 중앙에 마우스를 놓고 Enter")
    input()

    x1, y1 = pyautogui.position()

    print()
    print("② 보드 오른쪽 아래 칸의 중앙에 마우스를 놓고 Enter")
    input()

    x2, y2 = pyautogui.position()

    cell_x = (x2 - x1) / (COLS - 1)
    cell_y = (y2 - y1) / (ROWS - 1)

    if cell_x <= 5 or cell_y <= 5:
        raise RuntimeError(
            "보드 크기 계산 실패"
        )

    config["board_x"] = x1
    config["board_y"] = y1
    config["cell_x"] = cell_x
    config["cell_y"] = cell_y

    print()
    print("③ 조각 패널 왼쪽 위에 마우스를 놓고 Enter")
    input()

    px0, py0 = pyautogui.position()

    print()
    print("④ 조각 패널 오른쪽 아래에 마우스를 놓고 Enter")
    input()

    px1, py1 = pyautogui.position()

    config["panel_x0"] = min(px0, px1)
    config["panel_y0"] = min(py0, py1)
    config["panel_x1"] = max(px0, px1)
    config["panel_y1"] = max(py0, py1)

    print()
    print("⑤ 슬롯 1 회전 버튼에 마우스를 놓고 Enter")
    input()
    config["rotate_x"], config["rotate_y"][0] = pyautogui.position()

    print("⑥ 슬롯 2 회전 버튼")
    input()
    _, config["rotate_y"][1] = pyautogui.position()

    print("⑦ 슬롯 3 회전 버튼")
    input()
    _, config["rotate_y"][2] = pyautogui.position()

    print()
    print("⑧ 슬롯 1 뒤집기 버튼")
    input()
    _, config["flip_y"][0] = pyautogui.position()

    print("⑨ 슬롯 2 뒤집기 버튼")
    input()
    _, config["flip_y"][1] = pyautogui.position()

    print("⑩ 슬롯 3 뒤집기 버튼")
    input()
    _, config["flip_y"][2] = pyautogui.position()

    print()
    print("⑪ 한 칸 스킬 버튼")
    input()
    config["dot_button"] = list(
        pyautogui.position()
    )

    print()
    print("⑫ 바꿔 뽑기 버튼")
    input()
    config["swap_button"] = list(
        pyautogui.position()
    )

    save_config(config)

    print()
    print("보정 완료.")
    print(
        f"보드: {config['board_x']}, "
        f"{config['board_y']}"
    )
    print(
        f"셀: {config['cell_x']:.1f} x "
        f"{config['cell_y']:.1f}"
    )
    print()


# ============================================================
# 보드 인식
# ============================================================

def detect_board(img, config, debug=False):

    bx = config["board_x"]
    by = config["board_y"]

    cx = config["cell_x"]
    cy = config["cell_y"]

    board = [0] * ROWS

    for r in range(ROWS):

        for c in range(COLS):

            x = int(
                bx + c * cx
            )

            y = int(
                by + r * cy
            )

            x0 = max(
                0,
                int(x - cx * 0.30)
            )

            x1 = min(
                img.shape[1],
                int(x + cx * 0.30)
            )

            y0 = max(
                0,
                int(y - cy * 0.30)
            )

            y1 = min(
                img.shape[0],
                int(y + cy * 0.30)
            )

            patch = img[
                y0:y1,
                x0:x1
            ]

            if patch.size == 0:
                continue

            hsv = cv2.cvtColor(
                patch,
                cv2.COLOR_RGB2HSV
            )

            gray = cv2.cvtColor(
                patch,
                cv2.COLOR_RGB2GRAY
            )

            saturation = float(
                np.mean(hsv[:, :, 1])
            )

            std = float(
                np.std(gray)
            )

            # 실제 블록이 있는 칸은 보통
            # 빈 칸보다 색/질감 변화가 큼.
            value = (
                saturation * 0.65
                + std * 0.55
            )

            if value > 25:
                board[r] |= (1 << c)

    if debug:

        dbg = img.copy()

        for r in range(ROWS):

            for c in range(COLS):

                x = int(
                    bx + c * cx
                )

                y = int(
                    by + r * cy
                )

                if board[r] & (1 << c):
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 0)

                cv2.circle(
                    dbg,
                    (x, y),
                    5,
                    color,
                    -1
                )

        save_debug(
            "board_debug.png",
            dbg
        )

    return tuple(board)


# ============================================================
# 조각 인식
# ============================================================

def make_piece_mask(roi):

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_RGB2HSV
    )

    h, s, v = cv2.split(hsv)

    # 특정 색 하나가 아니라 넓게 잡는다.
    mask = (
        (s >= 55)
        &
        (v >= 50)
    ).astype(
        np.uint8
    ) * 255

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    return mask


def detect_piece_slot(
    img,
    config,
    slot,
    debug=False
):

    x0 = int(config["panel_x0"])
    x1 = int(config["panel_x1"])

    y0 = int(config["panel_y0"])
    y1 = int(config["panel_y1"])

    total_h = y1 - y0

    slot_h = total_h / 3.0

    sy0 = int(
        y0 + slot * slot_h
    )

    sy1 = int(
        y0 + (slot + 1) * slot_h
    )

    roi = img[
        sy0:sy1,
        x0:x1
    ]

    if roi.size == 0:
        return None

    mask = make_piece_mask(roi)

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            8
        )
    )

    candidates = []

    for i in range(1, count):

        bx, by, bw, bh, area = stats[i]

        if area < 35:
            continue

        if bw < 6 or bh < 6:
            continue

        if bw > roi.shape[1] * 0.9:
            continue

        if bh > roi.shape[0] * 0.9:
            continue

        candidates.append(
            (
                bx,
                by,
                bw,
                bh,
                area
            )
        )

    if not candidates:
        return None

    # 작은 컴포넌트 여러 개가 한 조각인 경우를
    # 전체 bounding box로 합친다.
    min_x = min(
        x for x, y, w, h, a in candidates
    )

    min_y = min(
        y for x, y, w, h, a in candidates
    )

    max_x = max(
        x + w
        for x, y, w, h, a in candidates
    )

    max_y = max(
        y + h
        for x, y, w, h, a in candidates
    )

    # 너무 작은 검출은 무시
    if max_x - min_x < 10:
        return None

    if max_y - min_y < 10:
        return None

    # 셀 크기를 기준으로 조각 형태 추정
    cell_x = max(
        8,
        config["cell_x"]
    )

    cell_y = max(
        8,
        config["cell_y"]
    )

    width = max_x - min_x
    height = max_y - min_y

    cols = max(
        1,
        min(
            6,
            int(
                round(
                    width / cell_x
                )
            )
        )
    )

    rows = max(
        1,
        min(
            6,
            int(
                round(
                    height / cell_y
                )
            )
        )
    )

    shape = []

    for r in range(rows):

        for c in range(cols):

            cx = int(
                min_x
                + (c + 0.5) * cell_x
            )

            cy = int(
                min_y
                + (r + 0.5) * cell_y
            )

            rx = max(
                3,
                int(cell_x * 0.30)
            )

            ry = max(
                3,
                int(cell_y * 0.30)
            )

            ax0 = max(
                0,
                cx - rx
            )

            ax1 = min(
                roi.shape[1],
                cx + rx
            )

            ay0 = max(
                0,
                cy - ry
            )

            ay1 = min(
                roi.shape[0],
                cy + ry
            )

            patch = mask[
                ay0:ay1,
                ax0:ax1
            ]

            if patch.size == 0:
                continue

            ratio = (
                np.count_nonzero(patch)
                /
                float(patch.size)
            )

            if ratio > 0.12:
                shape.append(
                    (r, c)
                )

    shape = normalize_shape(shape)

    if not shape:
        return None

    if len(shape) > 12:
        return None

    source_x = (
        x0
        + (min_x + max_x) // 2
    )

    source_y = (
        sy0
        + (min_y + max_y) // 2
    )

    confidence = min(
        1.0,
        0.4
        + len(shape) * 0.04
    )

    piece = Piece(
        slot=slot,
        cells=shape,
        source_xy=(
            source_x,
            source_y
        ),
        bbox=(
            x0 + min_x,
            sy0 + min_y,
            x0 + max_x,
            sy0 + max_y
        ),
        confidence=confidence
    )

    if debug:

        dbg = img.copy()

        bx1, by1, bx2, by2 = piece.bbox

        cv2.rectangle(
            dbg,
            (bx1, by1),
            (bx2, by2),
            (255, 0, 0),
            2
        )

        cv2.putText(
            dbg,
            f"S{slot+1} {piece.cells}",
            (bx1, max(20, by1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1
        )

        save_debug(
            f"piece_{slot+1}.png",
            dbg
        )

    return piece


def detect_pieces(
    img,
    config,
    debug=False
):

    result = []

    for slot in range(3):

        piece = detect_piece_slot(
            img,
            config,
            slot,
            debug
        )

        if piece:
            result.append(piece)

    return result


# ============================================================
# 마우스
# ============================================================

def click(x, y, dry_run=False):

    if dry_run:
        print(
            f"[DRY] CLICK {x},{y}"
        )
        return

    pyautogui.click(
        int(x),
        int(y)
    )


def rotate_or_flip(
    piece,
    target,
    config,
    dry_run=False
):

    sequence = transform_sequence(
        piece.cells,
        target
    )

    if sequence is None:
        return False

    slot = piece.slot

    for op in sequence:

        if op == "R":

            click(
                config["rotate_x"],
                config["rotate_y"][slot],
                dry_run
            )

        elif op == "F":

            click(
                config["rotate_x"],
                config["flip_y"][slot],
                dry_run
            )

        time.sleep(0.10)

    return True


def drag_piece(
    piece,
    target,
    row,
    col,
    config,
    dry_run=False
):

    bx = config["board_x"]
    by = config["board_y"]

    cx = config["cell_x"]
    cy = config["cell_y"]

    target_x = (
        bx
        + (col + 0.5) * cx
    )

    target_y = (
        by
        + (row + 0.5) * cy
    )

    sx, sy = piece.source_xy

    print(
        f"드래그 "
        f"S{piece.slot+1}: "
        f"({sx},{sy}) -> "
        f"({int(target_x)},{int(target_y)})"
    )

    if dry_run:
        return

    pyautogui.moveTo(
        sx,
        sy,
        duration=0.12
    )

    pyautogui.mouseDown()

    pyautogui.moveTo(
        target_x,
        target_y,
        duration=0.30
    )

    pyautogui.mouseUp()


# ============================================================
# 스킬
# ============================================================

def use_skill(
    config,
    name,
    dry_run=False
):

    if name == "dot":
        x, y = config["dot_button"]

    else:
        x, y = config["swap_button"]

    print(
        f"스킬 사용: {name}"
    )

    click(
        x,
        y,
        dry_run
    )

    time.sleep(0.45)


def one_gap(board):

    for r in range(ROWS):

        empty = (
            FULL_MASK
            ^ board[r]
        )

        if empty.bit_count() == 1:

            c = (
                empty
                & -empty
            ).bit_length() - 1

            return r, c

    return None


def has_clear_move(
    board,
    pieces
):

    for piece in pieces:

        for shape in orientations(
            piece.cells
        ):

            for _, _, _, cleared in legal_moves(
                board,
                shape
            ):

                if cleared:
                    return True

    return False


# ============================================================
# 핫키
# ============================================================

class Controller:

    def __init__(self):
        self.pause = False
        self.stop = False
        self.recalibrate = False


def install_hotkeys(ctrl):

    if keyboard is None:

        print(
            "keyboard 모듈 없음 - "
            "F8/F9/F6 사용 불가"
        )

        return

    keyboard.add_hotkey(
        "f8",
        lambda: setattr(
            ctrl,
            "pause",
            not ctrl.pause
        )
    )

    keyboard.add_hotkey(
        "f9",
        lambda: setattr(
            ctrl,
            "stop",
            True
        )
    )

    keyboard.add_hotkey(
        "f6",
        lambda: setattr(
            ctrl,
            "recalibrate",
            True
        )
    )


# ============================================================
# 출력
# ============================================================

def print_board(board):

    print()

    for r in range(ROWS):

        line = ""

        for c in range(COLS):

            if board[r] & (1 << c):
                line += "■"
            else:
                line += "·"

        print(
            f"{r:02d} {line}"
        )


def print_pieces(pieces):

    if not pieces:

        print(
            "조각: 없음"
        )

        return

    for p in pieces:

        print(
            f"S{p.slot+1} "
            f"{p.cells} "
            f"src={p.source_xy} "
            f"conf={p.confidence:.2f}"
        )


# ============================================================
# 메인
# ============================================================

def run(
    config,
    debug=False,
    dry_run=False
):

    ctrl = Controller()

    install_hotkeys(ctrl)

    print()
    print("=" * 60)
    print("한글 모아모아 봇")
    print("=" * 60)
    print()
    print("F8 : 일시정지 / 재개")
    print("F9 : 종료")
    print("F6 : 재보정")
    print()

    print(
        "5초 후 시작..."
    )

    for i in range(5, 0, -1):

        print(i)

        time.sleep(1)

    fail_count = 0

    while not ctrl.stop:

        if ctrl.pause:

            time.sleep(0.15)

            continue

        if ctrl.recalibrate:

            ctrl.recalibrate = False

            try:
                calibrate(config)

            except Exception:
                traceback.print_exc()

            continue

        img = screenshot()

        board = detect_board(
            img,
            config,
            debug
        )

        pieces = detect_pieces(
            img,
            config,
            debug
        )

        if debug:

            print_board(board)
            print_pieces(pieces)

        # ----------------------------------------------------
        # 조각 인식 실패
        # ----------------------------------------------------

        if not pieces:

            fail_count += 1

            if (
                fail_count == 1
                or fail_count % 10 == 0
            ):

                print(
                    f"조각 인식 실패 "
                    f"{fail_count}회"
                )

                if debug:

                    save_debug(
                        "piece_fail.png",
                        img
                    )

            time.sleep(0.25)

            continue

        fail_count = 0

        # ----------------------------------------------------
        # 최적 수 찾기
        # ----------------------------------------------------

        move = solve(
            board,
            pieces
        )

        # ----------------------------------------------------
        # 둘 곳 없음
        # ----------------------------------------------------

        if move is None:

            print(
                "현재 조각으로 배치 불가"
            )

            gap = one_gap(board)

            if (
                gap is not None
                and not has_clear_move(
                    board,
                    pieces
                )
            ):

                use_skill(
                    config,
                    "dot",
                    dry_run
                )

                continue

            use_skill(
                config,
                "swap",
                dry_run
            )

            continue

        slot, target_shape, row, col = move

        piece = None

        for p in pieces:

            if p.slot == slot:

                piece = p
                break

        if piece is None:

            print(
                "선택한 조각 재인식 실패"
            )

            time.sleep(0.2)

            continue

        print()
        print(
            f"선택 S{slot+1}"
        )

        print(
            f"모양: {target_shape}"
        )

        print(
            f"위치: row={row}, col={col}"
        )

        # ----------------------------------------------------
        # 회전 / 뒤집기
        # ----------------------------------------------------

        if not rotate_or_flip(
            piece,
            target_shape,
            config,
            dry_run
        ):

            print(
                "회전/반전 변환 실패"
            )

            continue

        # ----------------------------------------------------
        # 드래그
        # ----------------------------------------------------

        drag_piece(
            piece,
            target_shape,
            row,
            col,
            config,
            dry_run
        )

        time.sleep(0.40)

    print(
        "봇 종료"
    )


# ============================================================
# 시작
# ============================================================

def main():

    set_dpi_awareness()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--calibrate",
        action="store_true"
    )

    parser.add_argument(
        "--debug",
        action="store_true"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true"
    )

    args = parser.parse_args()

    config = load_config()

    if (
        args.calibrate
        or not CONFIG_FILE.exists()
    ):

        calibrate(config)

    run(
        config,
        debug=args.debug,
        dry_run=args.dry_run
    )


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "종료"
        )

    except Exception:

        print()
        print("=" * 60)
        print("오류 발생")
        print("=" * 60)

        traceback.print_exc()

        try:
            input(
                "\nEnter를 누르면 종료합니다..."
            )
        except Exception:
            pass
