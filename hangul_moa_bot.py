
import sys
import time
import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pyautogui
import keyboard

ROWS = 16
COLS = 10
FULL_MASK = (1 << COLS) - 1

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent

CONFIG_FILE = BASE_DIR / "hangul_moa_bot_config_v2.json"
DEBUG_DIR = BASE_DIR / "debug_v2"

pyautogui.PAUSE = 0.04
pyautogui.FAILSAFE = True

paused = False
stopped = False


@dataclass
class Piece:
    slot: int
    cells: Tuple[Tuple[int, int], ...]
    source_xy: Tuple[int, int]
    confidence: float


def log(msg: str):
    print(msg, flush=True)


def screenshot_rgb():
    return np.array(pyautogui.screenshot().convert("RGB"))


def normalize_shape(cells):
    if not cells:
        return tuple()
    min_r = min(r for r, c in cells)
    min_c = min(c for r, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def transform_shape(shape, rot=0, flip=False):
    pts = list(shape)
    if flip:
        pts = [(r, -c) for r, c in pts]
    for _ in range(rot % 4):
        pts = [(c, -r) for r, c in pts]
    return normalize_shape(pts)


def all_orientations(shape):
    out = []
    seen = set()
    for flip in (False, True):
        for rot in range(4):
            s = transform_shape(shape, rot, flip)
            if s not in seen:
                seen.add(s)
                out.append((s, flip, rot))
    return out


def shape_transform_sequence(current, target):
    target = normalize_shape(target)
    for flip in (False, True):
        for rot in range(4):
            if transform_shape(current, rot, flip) == target:
                return flip, rot
    return None


def place_piece(board, shape, top, left):
    rows = list(board)
    for dr, dc in shape:
        r = top + dr
        c = left + dc
        if r < 0 or r >= ROWS or c < 0 or c >= COLS:
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
    shape = normalize_shape(shape)
    if not shape:
        return []

    h = max(r for r, c in shape) + 1
    w = max(c for r, c in shape) + 1
    moves = []

    for r in range(ROWS - h + 1):
        for c in range(COLS - w + 1):
            result = place_piece(board, shape, r, c)
            if result is not None:
                new_board, cleared = result
                moves.append((r, c, cleared, new_board))
    return moves


def mobility(board, shapes):
    n = 0
    for shape in shapes:
        if legal_moves(board, shape):
            n += 1
    return n


def board_value(board):
    score = 0.0
    filled = 0
    for r in board:
        n = r.bit_count()
        filled += n
        score += n * n
        if n == 9:
            score += 35
        elif n == 8:
            score += 18
        elif n == 7:
            score += 8

    # Penalize isolated single holes in otherwise full rows.
    for r in board:
        if r == FULL_MASK:
            continue
        holes = COLS - r.bit_count()
        if holes == 1:
            score += 25

    return score + filled * 0.35


def solve(board, pieces):
    """
    Returns (slot, target_shape, row, col).
    Beam search is deliberately conservative. It only uses shapes that were
    actually detected from the screen.
    """
    if not pieces:
        return None

    initial = tuple(board)
    states = [(initial, 0.0, None, 0)]

    # Keep the beam small because the game UI is the bottleneck, not CPU.
    beam_width = 90

    for depth in range(min(3, len(pieces))):
        new_states = []

        for state_board, state_score, first_action, used_mask in states:
            for pi, piece in enumerate(pieces):
                if used_mask & (1 << pi):
                    continue

                for shape, flip, rot in all_orientations(piece.cells):
                    for r, c, cleared, new_board in legal_moves(state_board, shape):
                        immediate = (
                            cleared * 1200
                            + board_value(new_board)
                            + mobility(
                                new_board,
                                [
                                    p.cells
                                    for j, p in enumerate(pieces)
                                    if j != pi and not (used_mask & (1 << j))
                                ],
                            ) * 8
                        )

                        total = state_score + immediate
                        action = first_action
                        if action is None:
                            action = (piece.slot, shape, r, c)

                        new_states.append(
                            (new_board, total, action, used_mask | (1 << pi))
                        )

        if not new_states:
            break

        new_states.sort(key=lambda x: x[1], reverse=True)
        states = new_states[:beam_width]

    if states:
        best = max(states, key=lambda x: x[1])
        if best[2] is not None:
            return best[2]

    # Absolute fallback: if the board is empty, any detected shape has a legal
    # placement. Never enter an endless swap loop because of a solver failure.
    if sum(x.bit_count() for x in board) == 0:
        p = pieces[0]
        moves = legal_moves(board, p.cells)
        if moves:
            r, c, _, _ = moves[0]
            return p.slot, p.cells, r, c

    return None


def median_rgb(img, cx, cy, rx, ry):
    h, w = img.shape[:2]
    x0 = max(0, int(cx - rx))
    x1 = min(w, int(cx + rx + 1))
    y0 = max(0, int(cy - ry))
    y1 = min(h, int(cy + ry + 1))
    patch = img[y0:y1, x0:x1]
    if patch.size == 0:
        return np.array([0, 0, 0], dtype=np.float32)
    return np.median(patch.reshape(-1, 3), axis=0).astype(np.float32)


def cell_centers(cfg):
    bx = cfg["board_x"]
    by = cfg["board_y"]
    cx = cfg["cell_x"]
    cy = cfg["cell_y"]
    for r in range(ROWS):
        for c in range(COLS):
            yield r, c, (bx + (c + 0.5) * cx, by + (r + 0.5) * cy)


def capture_empty_reference(img, cfg):
    ref = []
    for r, c, (x, y) in cell_centers(cfg):
        ref.append(median_rgb(img, x, y, cfg["cell_x"] * 0.24, cfg["cell_y"] * 0.24))
    return [v.tolist() for v in ref]


def board_from_reference(img, cfg, threshold=None):
    refs = cfg.get("empty_reference")
    if not refs or len(refs) != ROWS * COLS:
        return board_from_fallback(img, cfg)

    ref = np.array(refs, dtype=np.float32).reshape(ROWS, COLS, 3)

    # Compare in Lab because RGB distance is more sensitive to brightness.
    ref_lab = cv2.cvtColor(ref.reshape(1, ROWS * COLS, 3).astype(np.uint8),
                           cv2.COLOR_RGB2LAB).reshape(ROWS, COLS, 3).astype(np.float32)

    samples = []
    for r, c, (x, y) in cell_centers(cfg):
        samples.append(median_rgb(
            img, x, y, cfg["cell_x"] * 0.24, cfg["cell_y"] * 0.24
        ))
    cur = np.array(samples, dtype=np.float32).reshape(ROWS, COLS, 3)
    cur_lab = cv2.cvtColor(cur.reshape(1, ROWS * COLS, 3).astype(np.uint8),
                           cv2.COLOR_RGB2LAB).reshape(ROWS, COLS, 3).astype(np.float32)

    dist = np.linalg.norm(cur_lab - ref_lab, axis=2)

    if threshold is None:
        threshold = float(cfg.get("board_diff_threshold", 18.0))

    board = []
    for r in range(ROWS):
        mask = 0
        for c in range(COLS):
            if dist[r, c] >= threshold:
                mask |= (1 << c)
        board.append(mask)

    return tuple(board), dist


def board_from_fallback(img, cfg):
    # Emergency fallback only. The calibrated reference method is preferred.
    board = []
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    for r in range(ROWS):
        mask = 0
        for c in range(COLS):
            x = int(cfg["board_x"] + (c + 0.5) * cfg["cell_x"])
            y = int(cfg["board_y"] + (r + 0.5) * cfg["cell_y"])
            x0 = max(0, int(x - cfg["cell_x"] * 0.22))
            x1 = min(hsv.shape[1], int(x + cfg["cell_x"] * 0.22))
            y0 = max(0, int(y - cfg["cell_y"] * 0.22))
            y1 = min(hsv.shape[0], int(y + cfg["cell_y"] * 0.22))
            p = hsv[y0:y1, x0:x1]
            if p.size == 0:
                continue
            # Colored blocks are much more saturated than the empty blue cell.
            sat = float(np.median(p[:, :, 1]))
            if sat > 155:
                mask |= (1 << c)
        board.append(mask)

    return tuple(board), None


def save_board_debug(img, cfg, board, dist=None):
    DEBUG_DIR.mkdir(exist_ok=True)
    out = img.copy()

    for r, c, (x, y) in cell_centers(cfg):
        x = int(x)
        y = int(y)
        if board[r] & (1 << c):
            cv2.rectangle(
                out,
                (int(x - cfg["cell_x"] * 0.45), int(y - cfg["cell_y"] * 0.45)),
                (int(x + cfg["cell_x"] * 0.45), int(y + cfg["cell_y"] * 0.45)),
                (255, 60, 60),
                1,
            )

    cv2.imwrite(
        str(DEBUG_DIR / "board_debug_v2.png"),
        cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
    )

    with open(DEBUG_DIR / "board_debug_v2.txt", "w", encoding="utf-8") as f:
        for r in range(ROWS):
            s = "".join("#" if board[r] & (1 << c) else "." for c in range(COLS))
            f.write(f"{r:02d} {s}\n")

        if dist is not None:
            f.write("\nLab distance:\n")
            for r in range(ROWS):
                f.write(" ".join(f"{dist[r,c]:.1f}" for c in range(COLS)) + "\n")


def piece_mask(img, center):
    x, y = center
    half_w = 34
    half_h = 30

    x0 = max(0, int(x - half_w))
    x1 = min(img.shape[1], int(x + half_w))
    y0 = max(0, int(y - half_h))
    y1 = min(img.shape[0], int(y + half_h))

    roi = img[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Game piece colors are saturated. The slot background is light and the
    # blue game UI is around hue 90~125, so exclude that blue range.
    mask = (
        (s >= 75)
        & (v >= 85)
        & ~((h >= 85) & (h <= 125))
    ).astype(np.uint8) * 255

    # Remove tiny text/icon noise.
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask, (x0, y0)


def score_grid_candidate(mask, p, ox, oy):
    h, w = mask.shape
    x1 = min(w, ox + 7 * p)
    y1 = min(h, oy + 7 * p)

    if ox < 0 or oy < 0 or x1 - ox < p or y1 - oy < p:
        return None

    ratios = []
    coords = []

    for rr in range(7):
        for cc in range(7):
            xa = ox + cc * p
            ya = oy + rr * p
            xb = xa + p
            yb = ya + p
            if xb > w or yb > h:
                continue
            patch = mask[ya:yb, xa:xb]
            ratios.append(float(np.count_nonzero(patch)) / float(p * p))
            coords.append((rr, cc))

    if not ratios:
        return None

    arr = np.array(ratios, dtype=np.float32)

    # Actual preview cells contain a fairly large colored fraction.
    occ = arr >= 0.22
    count = int(occ.sum())

    if count < 1 or count > 12:
        return None

    # Penalize candidates that classify almost everything as a piece.
    occupied_mean = float(arr[occ].mean()) if np.any(occ) else 0.0
    empty_mean = float(arr[~occ].mean()) if np.any(~occ) else 0.0
    separation = occupied_mean - empty_mean

    # Require a connected shape in the inferred grid.
    grid = np.zeros((7, 7), np.uint8)
    for i, is_occ in enumerate(occ):
        if is_occ:
            r, c = coords[i]
            grid[r, c] = 1

    n, _, stats, _ = cv2.connectedComponentsWithStats(grid, 4)
    largest = max((stats[i, 4] for i in range(1, n)), default=0)
    connected_ratio = largest / max(1, count)

    # Prefer pitches that create a clean separated shape.
    score = separation * 100 + connected_ratio * 25 - abs(count - 4) * 0.5
    return score, grid


def detect_piece(img, cfg, slot):
    centers = cfg["piece_centers"]
    center = centers[slot]

    mask, origin = piece_mask(img, center)
    DEBUG_DIR.mkdir(exist_ok=True)
    cv2.imwrite(
        str(DEBUG_DIR / f"piece_{slot+1}_v2_mask.png"),
        mask,
    )

    best = None
    # Preview block size on the actual UI is much smaller than board cells.
    for p in range(7, 15):
        for ox in range(0, max(1, mask.shape[1] - p * 2)):
            for oy in range(0, max(1, mask.shape[0] - p * 2)):
                result = score_grid_candidate(mask, p, ox, oy)
                if result is None:
                    continue
                score, grid = result
                if best is None or score > best[0]:
                    best = (score, grid, p, ox, oy)

    if best is None:
        return None

    score, grid, p, ox, oy = best

    cells = []
    screen_points = []

    for r in range(7):
        for c in range(7):
            if grid[r, c]:
                cells.append((r, c))
                sx = origin[0] + ox + (c + 0.5) * p
                sy = origin[1] + oy + (r + 0.5) * p
                screen_points.append((sx, sy))

    if not cells:
        return None

    cells = normalize_shape(cells)

    # Recalculate centroid after normalization.
    min_r = min(r for r, c in cells)
    min_c = min(c for r, c in cells)
    pts = []
    for r, c in cells:
        pts.append((
            origin[0] + ox + (c - min_c + 0.5) * p,
            origin[1] + oy + (r - min_r + 0.5) * p,
        ))

    sx = int(round(sum(x for x, y in pts) / len(pts)))
    sy = int(round(sum(y for x, y in pts) / len(pts)))

    confidence = min(1.0, max(0.0, (score - 15.0) / 100.0))

    if confidence < 0.18:
        return None

    return Piece(
        slot=slot,
        cells=tuple(cells),
        source_xy=(sx, sy),
        confidence=confidence,
    )


def detect_pieces(img, cfg):
    pieces = []
    for slot in range(3):
        p = detect_piece(img, cfg, slot)
        if p is not None:
            pieces.append(p)
            log(
                f"S{slot+1} shape={p.cells} src={p.source_xy} "
                f"conf={p.confidence:.2f}"
            )
    return pieces


def board_count(board):
    return sum(x.bit_count() for x in board)


def wait_enter(prompt):
    input(prompt + "\n> ")


def calibration_click(label):
    wait_enter(
        f"[CAL] {label}\n"
        "마우스를 해당 위치에 올려놓고 Enter를 누르세요."
    )
    pos = pyautogui.position()
    log(f"  -> {pos}")
    return [int(pos.x), int(pos.y)]


def calibrate():
    log("")
    log("========================================")
    log(" HangulMoaBot V2 CALIBRATION")
    log("========================================")
    log("")
    log("주의: 캘리브레이션 중 게임판은 반드시 '완전히 빈 상태'여야 합니다.")
    log("게임 창은 움직이지 말고 고정하세요.")
    log("")

    tl = calibration_click("1/12 보드의 왼쪽 위 칸 '중앙'")
    br = calibration_click("2/12 보드의 오른쪽 아래 칸 '중앙'")

    cell_x = (br[0] - tl[0]) / (COLS - 1)
    cell_y = (br[1] - tl[1]) / (ROWS - 1)

    # tl/br are cell centers.
    board_x = tl[0] - cell_x / 2
    board_y = tl[1] - cell_y / 2

    p1 = calibration_click("3/12 첫 번째 보유 조각의 '가운데'")
    p2 = calibration_click("4/12 두 번째 보유 조각의 '가운데'")
    p3 = calibration_click("5/12 세 번째 보유 조각의 '가운데'")

    r1 = calibration_click("6/12 첫 번째 조각의 '회전' 버튼")
    f1 = calibration_click("7/12 첫 번째 조각의 '반전' 버튼")
    r2 = calibration_click("8/12 두 번째 조각의 '회전' 버튼")
    f2 = calibration_click("9/12 두 번째 조각의 '반전' 버튼")
    r3 = calibration_click("10/12 세 번째 조각의 '회전' 버튼")
    f3 = calibration_click("11/12 세 번째 조각의 '반전' 버튼")
    swap = calibration_click("12/12 '바꿔 뽑기' 버튼")

    cfg = {
        "board_x": float(board_x),
        "board_y": float(board_y),
        "cell_x": float(cell_x),
        "cell_y": float(cell_y),
        "piece_centers": [p1, p2, p3],
        "rotate_buttons": [r1, r2, r3],
        "flip_buttons": [f1, f2, f3],
        "swap_button": swap,
        "board_diff_threshold": 18.0,
    }

    log("")
    log("보드가 완전히 비어 있는지 다시 확인하세요.")
    wait_enter("빈 보드 상태에서 Enter를 누르면 빈 보드 기준색을 저장합니다.")

    img = screenshot_rgb()
    cfg["empty_reference"] = capture_empty_reference(img, cfg)

    CONFIG_FILE.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    log("")
    log(f"캘리브레이션 저장 완료: {CONFIG_FILE}")
    log(
        f"보드: x={cfg['board_x']:.1f}, y={cfg['board_y']:.1f}, "
        f"cell={cfg['cell_x']:.2f}x{cfg['cell_y']:.2f}"
    )
    log("F6으로 언제든 다시 캘리브레이션할 수 있습니다.")
    return cfg


def load_config():
    if not CONFIG_FILE.exists():
        return calibrate()

    try:
        cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        required = [
            "board_x", "board_y", "cell_x", "cell_y",
            "piece_centers", "rotate_buttons", "flip_buttons",
            "swap_button", "empty_reference",
        ]
        if any(k not in cfg for k in required):
            raise ValueError("old/incomplete config")
        return cfg
    except Exception as e:
        log(f"기존 설정을 읽지 못했습니다: {e}")
        return calibrate()


def press_button(pos):
    pyautogui.click(int(pos[0]), int(pos[1]))
    time.sleep(0.22)


def orient_piece(piece, target_shape, cfg):
    target_shape = normalize_shape(target_shape)
    current = piece.cells

    seq = shape_transform_sequence(current, target_shape)
    if seq is None:
        return False

    flip, rot = seq
    slot = piece.slot

    # Apply flip first, then rotation.
    if flip:
        press_button(cfg["flip_buttons"][slot])

    for _ in range(rot):
        press_button(cfg["rotate_buttons"][slot])

    time.sleep(0.25)
    return True


def target_centroid(shape, row, col, cfg):
    mean_r = sum(r for r, c in shape) / len(shape)
    mean_c = sum(c for r, c in shape) / len(shape)

    x = cfg["board_x"] + (col + mean_c + 0.5) * cfg["cell_x"]
    y = cfg["board_y"] + (row + mean_r + 0.5) * cfg["cell_y"]
    return int(round(x)), int(round(y))


def drag_piece(piece, shape, row, col, cfg):
    sx, sy = piece.source_xy
    tx, ty = target_centroid(shape, row, col, cfg)

    log(f"DRAG S{piece.slot+1}: ({sx},{sy}) -> ({tx},{ty})")

    pyautogui.moveTo(sx, sy, duration=0.12)
    time.sleep(0.08)
    pyautogui.mouseDown()
    time.sleep(0.12)
    pyautogui.moveTo(tx, ty, duration=0.42)
    time.sleep(0.15)
    pyautogui.mouseUp()
    time.sleep(0.55)


def board_changed(before, after):
    if before is None or after is None:
        return False
    return any(a != b for a, b in zip(before, after))


def verify_drop(before_board, cfg):
    for _ in range(4):
        time.sleep(0.20)
        img = screenshot_rgb()
        after, _ = board_from_reference(img, cfg)
        if board_changed(before_board, after):
            return True, after
    return False, after


def safe_place(piece, target_shape, row, col, before_board, cfg):
    # First attempt.
    if not orient_piece(piece, target_shape, cfg):
        log("회전/반전 조합을 찾지 못함")
        return False, before_board

    # Re-detect the piece after orientation. This also gives us a corrected
    # screen centroid if the UI animation moved the preview slightly.
    img = screenshot_rgb()
    redetected = detect_piece(img, cfg, piece.slot)
    if redetected is not None:
        piece = redetected

    # Try the calculated target and four small offsets.
    base_x, base_y = target_centroid(target_shape, row, col, cfg)
    offsets = [
        (0, 0),
        (0.18, 0),
        (-0.18, 0),
        (0, 0.18),
        (0, -0.18),
    ]

    for ox, oy in offsets:
        tx = int(round(base_x + ox * cfg["cell_x"]))
        ty = int(round(base_y + oy * cfg["cell_y"]))

        sx, sy = piece.source_xy
        log(f"DROP try: ({sx},{sy}) -> ({tx},{ty})")

        pyautogui.moveTo(sx, sy, duration=0.12)
        pyautogui.mouseDown()
        time.sleep(0.12)
        pyautogui.moveTo(tx, ty, duration=0.42)
        time.sleep(0.12)
        pyautogui.mouseUp()

        time.sleep(0.60)
        img2 = screenshot_rgb()
        after, _ = board_from_reference(img2, cfg)

        if board_changed(before_board, after):
            log("배치 성공: 보드 변화 확인")
            return True, after

    log("배치 실패: 보드 변화가 없음")
    return False, before_board


def save_piece_debug(img, slot):
    DEBUG_DIR.mkdir(exist_ok=True)
    cv2.imwrite(
        str(DEBUG_DIR / f"piece_{slot+1}_v2.png"),
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
    )


def hotkeys():
    global paused, stopped

    def toggle():
        global paused
        paused = not paused
        log("PAUSE" if paused else "RESUME")

    def stop():
        global stopped
        stopped = True
        log("STOP")

    def recal():
        log("F6: recalibration")
        try:
            calibrate()
        except Exception as e:
            log(f"재보정 실패: {e}")

    keyboard.add_hotkey("f8", toggle)
    keyboard.add_hotkey("f9", stop)
    keyboard.add_hotkey("f6", recal)


def main():
    global stopped

    log("HangulMoaBot V2 시작")
    log("F8 pause/resume | F9 stop | F6 recalibrate")
    log(f"설정: {CONFIG_FILE}")

    cfg = load_config()
    hotkeys()

    # Give the user time to move the console out of the game.
    log("")
    log("게임 화면과 보유 조각이 모두 보이는지 확인하세요.")
    log("5초 후 시작합니다.")
    time.sleep(5)

    consecutive_failures = 0
    consecutive_swaps = 0

    while not stopped:
        if paused:
            time.sleep(0.2)
            continue

        try:
            img = screenshot_rgb()

            board, dist = board_from_reference(img, cfg)
            count = board_count(board)

            pieces = detect_pieces(img, cfg)

            save_board_debug(img, cfg, board, dist)

            log(f"BOARD occupied={count}/160 | pieces={len(pieces)}")

            # Critical safety rule: an empty board must never be treated as
            # "no legal move".
            if count == 0 and not pieces:
                log("빈 보드인데 조각 인식 실패 -> swap 금지, 재탐색")
                time.sleep(0.5)
                continue

            if not pieces:
                log("조각 인식 실패 -> swap 하지 않고 재탐색")
                time.sleep(0.5)
                continue

            move = solve(board, pieces)

            if move is None:
                # Never endlessly reroll on an empty board.
                if count == 0:
                    p = pieces[0]
                    moves = legal_moves(board, p.cells)
                    if moves:
                        r, c, _, _ = moves[0]
                        move = (p.slot, p.cells, r, c)

                if move is None:
                    consecutive_swaps += 1
                    log(f"현재 조각으로 배치 불가 -> swap ({consecutive_swaps}/3)")

                    if consecutive_swaps > 3:
                        log("swap 3회 초과. 인식 오류 가능성이 있어 안전 정지합니다.")
                        break

                    press_button(cfg["swap_button"])
                    time.sleep(0.8)
                    continue

            consecutive_swaps = 0

            slot, target_shape, row, col = move
            selected = next((p for p in pieces if p.slot == slot), None)

            if selected is None:
                log("선택 조각이 사라짐 -> 재탐색")
                continue

            log(
                f"TARGET S{slot+1} shape={target_shape} "
                f"at row={row}, col={col}"
            )

            ok, new_board = safe_place(
                selected, target_shape, row, col, board, cfg
            )

            if ok:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                log(f"배치 실패 {consecutive_failures}/3")

                if consecutive_failures >= 3:
                    log("연속 3회 배치 실패. debug_v2를 확인하고 안전 정지합니다.")
                    break

            time.sleep(0.25)

        log("HangulMoaBot V2 종료")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        log(f"FATAL: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        input("Enter를 누르면 종료합니다.")
