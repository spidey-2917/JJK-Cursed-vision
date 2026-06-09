import numpy as np
import cv2
import math
import random
import time

WRIST = 0
THUMB_TIP = 4
THUMB_IP = 3
THUMB_MCP = 2
THUMB_CMC = 1
INDEX_TIP = 8
INDEX_DIP = 7
INDEX_PIP = 6
INDEX_MCP = 5
MIDDLE_TIP = 12
MIDDLE_DIP = 11
MIDDLE_PIP = 10
MIDDLE_MCP = 9
RING_TIP = 16
RING_DIP = 15
RING_PIP = 14
RING_MCP = 13
PINKY_TIP = 20
PINKY_DIP = 19
PINKY_PIP = 18
PINKY_MCP = 17

GESTURE_COLORS = {
    "DOMAIN EXPANSION": (255, 140, 0),
    "CURSED TECHNIQUE": (60, 0, 220),
    "ENERGY RELEASE": (0, 220, 80),
    "HOLLOW PURPLE": (180, 0, 255),
    "CLEAVE": (0, 180, 255),
}

GESTURE_SUBTITLES = {
    "DOMAIN EXPANSION": "Infinite Void",
    "CURSED TECHNIQUE": "Malevolent Shrine",
    "ENERGY RELEASE": "Divergent Fist",
    "HOLLOW PURPLE": "Imaginary Technique",
    "CLEAVE": "Dismantle",
}

GESTURE_KANJI = {
    "DOMAIN EXPANSION": "\u7121\u91cf\u7a7a\u51e6",
    "CURSED TECHNIQUE": "\u546a\u8853\u5efb\u6226",
    "ENERGY RELEASE": "\u767a\u6563\u62f3",
    "HOLLOW PURPLE": "\u865a\u5f0f\u8336\u7d2b",
    "CLEAVE": "\u89e3",
}


def _dist(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def _finger_up(landmarks, tip, pip):
    return landmarks[tip][1] < landmarks[pip][1]


def _thumb_up(landmarks):
    tip_x = landmarks[THUMB_TIP][0]
    mcp_x = landmarks[THUMB_MCP][0]
    wrist_x = landmarks[WRIST][0]
    return tip_x < mcp_x if wrist_x > mcp_x else tip_x > mcp_x


def _finger_states(landmarks):
    return [
        _thumb_up(landmarks),
        _finger_up(landmarks, INDEX_TIP, INDEX_PIP),
        _finger_up(landmarks, MIDDLE_TIP, MIDDLE_PIP),
        _finger_up(landmarks, RING_TIP, RING_PIP),
        _finger_up(landmarks, PINKY_TIP, PINKY_PIP),
    ]


def _is_domain_expansion(landmarks):
    index_up = _finger_up(landmarks, INDEX_TIP, INDEX_PIP)
    middle_up = _finger_up(landmarks, MIDDLE_TIP, MIDDLE_PIP)
    ring_down = not _finger_up(landmarks, RING_TIP, RING_PIP)
    pinky_down = not _finger_up(landmarks, PINKY_TIP, PINKY_PIP)
    spread = _dist(landmarks[INDEX_TIP][:2], landmarks[MIDDLE_TIP][:2]) > 0.04
    return index_up and middle_up and ring_down and pinky_down and spread


def _is_hollow_purple(landmarks):
    fingers = _finger_states(landmarks)
    all_up = all(fingers[1:])
    thumb_up = fingers[0]
    palm_open = _dist(landmarks[INDEX_TIP][:2], landmarks[PINKY_TIP][:2]) > 0.12
    return all_up and thumb_up and palm_open


def _is_cleave(landmarks):
    index_up = _finger_up(landmarks, INDEX_TIP, INDEX_PIP)
    middle_up = _finger_up(landmarks, MIDDLE_TIP, MIDDLE_PIP)
    ring_up = _finger_up(landmarks, RING_TIP, RING_PIP)
    pinky_down = not _finger_up(landmarks, PINKY_TIP, PINKY_PIP)
    close = _dist(landmarks[INDEX_TIP][:2], landmarks[MIDDLE_TIP][:2]) < 0.03
    return index_up and middle_up and ring_up and pinky_down and close


def _is_cursed_technique(landmarks):
    fingers = _finger_states(landmarks)
    count = sum(fingers)
    if count <= 1:
        return True
    if _dist(landmarks[THUMB_TIP][:2], landmarks[INDEX_TIP][:2]) < 0.06:
        return True
    return False


def _is_energy_release(landmarks):
    fingers = _finger_states(landmarks)
    return fingers[1] and fingers[2] and fingers[3] and fingers[4]


def detect_gesture(landmarks):
    if _is_domain_expansion(landmarks):
        return "DOMAIN EXPANSION"
    if _is_hollow_purple(landmarks):
        return "HOLLOW PURPLE"
    if _is_cleave(landmarks):
        return "CLEAVE"
    if _is_cursed_technique(landmarks):
        return "CURSED TECHNIQUE"
    if _is_energy_release(landmarks):
        return "ENERGY RELEASE"
    return None


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "decay", "radius", "color", "gravity")

    def __init__(self, x, y, color):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 9)
        self.x = float(x)
        self.y = float(y)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = 1.0
        self.decay = random.uniform(0.02, 0.06)
        self.radius = random.randint(2, 6)
        self.color = color
        self.gravity = random.uniform(0.05, 0.15)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        self.vx *= 0.97
        self.life -= self.decay

    def alive(self):
        return self.life > 0

    def draw(self, frame):
        if not self.alive():
            return
        alpha = max(0.0, self.life)
        r = max(1, int(self.radius * alpha))
        cx, cy = int(self.x), int(self.y)
        h, w = frame.shape[:2]
        if 0 <= cx < w and 0 <= cy < h:
            overlay = frame.copy()
            cv2.circle(overlay, (cx, cy), r, self.color, -1, cv2.LINE_AA)
            blend = alpha * 0.9
            cv2.addWeighted(overlay, blend, frame, 1 - blend, 0, frame)


class LightningBolt:
    __slots__ = ("color", "life", "decay", "points")

    def __init__(self, start, end, color, segments=8, jaggedness=15):
        self.color = color
        self.life = 1.0
        self.decay = 0.08
        self.points = self._generate(start, end, segments, jaggedness)

    @staticmethod
    def _generate(start, end, segments, jaggedness):
        pts = [start]
        for i in range(1, segments):
            t = i / segments
            mx = int(start[0] + (end[0] - start[0]) * t + random.randint(-jaggedness, jaggedness))
            my = int(start[1] + (end[1] - start[1]) * t + random.randint(-jaggedness, jaggedness))
            pts.append((mx, my))
        pts.append(end)
        return pts

    def update(self):
        self.life -= self.decay
        if self.life > 0:
            self.points = self._generate(self.points[0], self.points[-1], 8, 15)

    def alive(self):
        return self.life > 0

    def draw(self, frame):
        if not self.alive():
            return
        alpha = max(0.0, self.life)
        overlay = frame.copy()
        for i in range(len(self.points) - 1):
            cv2.line(overlay, self.points[i], self.points[i + 1],
                     self.color, max(1, int(3 * alpha)), cv2.LINE_AA)
            cv2.line(overlay, self.points[i], self.points[i + 1],
                     (255, 255, 255), 1, cv2.LINE_AA)
        blend = alpha * 0.85
        cv2.addWeighted(overlay, blend, frame, 1 - blend, 0, frame)


class EffectState:
    def __init__(self):
        self.radius = 0
        self.max_radius = 140
        self.speed = 4
        self.alpha = 1.0
        self.particles = []
        self.lightning_bolts = []
        self.flash_alpha = 0.0
        self.flash_color = (255, 255, 255)
        self.text_scale = 0.0
        self.text_alpha = 0.0
        self.text_target_scale = 1.0
        self.aura_phase = 0.0
        self.shockwaves = []
        self.last_gesture = None
        self.gesture_frame = 0
        self.trigger_cooldown = 0

    def trigger(self, gesture, hand_center, color):
        self.flash_alpha = 0.45
        self.flash_color = color
        self.text_scale = 0.3
        self.text_alpha = 1.0

        cx, cy = hand_center
        for _ in range(50):
            self.particles.append(Particle(cx, cy, color))

        self.shockwaves.append({
            "r": 0, "max_r": 200, "speed": 8,
            "alpha": 1.0, "color": color, "cx": cx, "cy": cy,
        })
        self.trigger_cooldown = 25

    def update(self, gesture, hand_center, color):
        self.radius += self.speed
        self.alpha = 1.0 - (self.radius / self.max_radius)
        if self.radius >= self.max_radius:
            self.radius = 0
            self.alpha = 1.0

        self.aura_phase = (self.aura_phase + 0.08) % (2 * math.pi)
        self.flash_alpha = max(0.0, self.flash_alpha - 0.04)

        if self.text_scale < self.text_target_scale:
            self.text_scale = min(self.text_target_scale, self.text_scale + 0.06)
        self.text_alpha = max(0.0, self.text_alpha - 0.008)

        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.alive()]

        if hand_center and gesture and self.trigger_cooldown == 0:
            cx, cy = hand_center
            if random.random() < 0.4:
                self.particles.append(Particle(
                    cx + random.randint(-30, 30),
                    cy + random.randint(-30, 30),
                    color,
                ))

        if hand_center and gesture and random.random() < 0.08:
            cx, cy = hand_center
            angle = random.uniform(0, 2 * math.pi)
            dist = random.randint(40, 100)
            ex = int(cx + math.cos(angle) * dist)
            ey = int(cy + math.sin(angle) * dist)
            self.lightning_bolts.append(LightningBolt((cx, cy), (ex, ey), color))

        for bolt in self.lightning_bolts:
            bolt.update()
        self.lightning_bolts = [b for b in self.lightning_bolts if b.alive()]

        for sw in self.shockwaves:
            sw["r"] += sw["speed"]
            sw["alpha"] = max(0.0, 1.0 - sw["r"] / sw["max_r"])
        self.shockwaves = [sw for sw in self.shockwaves if sw["alpha"] > 0]

        if self.trigger_cooldown > 0:
            self.trigger_cooldown -= 1
            if hand_center:
                cx, cy = hand_center
                for _ in range(3):
                    self.particles.append(Particle(cx, cy, color))

        self.gesture_frame += 1


def _draw_aura(frame, gesture, hand_center, effect_state):
    if not hand_center or gesture not in GESTURE_COLORS:
        return
    color = GESTURE_COLORS[gesture]
    cx, cy = hand_center
    pulse = math.sin(effect_state.aura_phase)

    for i in range(3):
        r = int(55 + i * 22 + pulse * (8 + i * 4))
        a = max(0.0, 0.18 - i * 0.05)
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), r, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, a, frame, 1 - a, 0, frame)

    ring_r = int(60 + pulse * 8)
    cv2.circle(frame, (cx, cy), ring_r, color, 2, cv2.LINE_AA)


def _draw_shockwaves(frame, effect_state):
    for sw in effect_state.shockwaves:
        overlay = frame.copy()
        thickness = max(1, int(4 * sw["alpha"]))
        cv2.circle(overlay, (sw["cx"], sw["cy"]), sw["r"], sw["color"], thickness, cv2.LINE_AA)
        if sw["r"] > 20:
            cv2.circle(overlay, (sw["cx"], sw["cy"]), sw["r"] - 15, sw["color"], 1, cv2.LINE_AA)
        blend = sw["alpha"] * 0.85
        cv2.addWeighted(overlay, blend, frame, 1 - blend, 0, frame)


def _draw_screen_flash(frame, effect_state):
    if effect_state.flash_alpha <= 0:
        return
    overlay = frame.copy()
    overlay[:] = effect_state.flash_color
    cv2.addWeighted(overlay, effect_state.flash_alpha, frame,
                    1 - effect_state.flash_alpha, 0, frame)


def _draw_particles(frame, effect_state):
    for p in effect_state.particles:
        p.draw(frame)


def _draw_lightning(frame, effect_state):
    for bolt in effect_state.lightning_bolts:
        bolt.draw(frame)


def draw_gesture_effects(frame, gesture, hand_center, effect_state):
    if gesture not in GESTURE_COLORS:
        return

    color = GESTURE_COLORS[gesture]

    if gesture != effect_state.last_gesture:
        effect_state.trigger(gesture, hand_center, color)
        effect_state.last_gesture = gesture
        effect_state.radius = 0

    effect_state.update(gesture, hand_center, color)

    _draw_aura(frame, gesture, hand_center, effect_state)
    _draw_shockwaves(frame, effect_state)

    cx, cy = hand_center
    ring_alpha = max(0.0, effect_state.alpha)
    if effect_state.radius > 0:
        overlay = frame.copy()
        thickness = max(2, int(4 * ring_alpha))
        cv2.circle(overlay, (cx, cy), effect_state.radius, color, thickness, cv2.LINE_AA)
        blend = ring_alpha * 0.9
        cv2.addWeighted(overlay, blend, frame, 1 - blend, 0, frame)
        inner_r = max(0, effect_state.radius - 25)
        if inner_r > 0:
            cv2.circle(frame, (cx, cy), inner_r, color, 1, cv2.LINE_AA)

    _draw_particles(frame, effect_state)
    _draw_lightning(frame, effect_state)
    _draw_screen_flash(frame, effect_state)


def draw_gesture_label(frame, gesture, effect_state=None):
    if gesture not in GESTURE_COLORS:
        return

    color = GESTURE_COLORS[gesture]
    subtitle = GESTURE_SUBTITLES.get(gesture, "")
    h, w = frame.shape[:2]

    banner_h = 90
    banner_overlay = frame.copy()
    for i in range(banner_h):
        y_pos = h - banner_h + i
        cv2.line(banner_overlay, (0, y_pos), (w, y_pos), (0, 0, 0), 1)
    cv2.addWeighted(banner_overlay, 0.7, frame, 0.3, 0, frame)

    cv2.line(frame, (0, h - banner_h), (w, h - banner_h), color, 2)
    cv2.line(frame, (0, h - banner_h + 4), (w, h - banner_h + 4), color, 1)

    scale = effect_state.text_scale if effect_state else 1.0
    scale = max(0.3, min(1.0, scale))

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.3 * scale
    thickness = 2
    text_size = cv2.getTextSize(gesture, font, font_scale, thickness)[0]
    tx = (w - text_size[0]) // 2
    ty = h - banner_h + 38

    for offset, alpha_mul in [(4, 0.3), (2, 0.5)]:
        shadow = frame.copy()
        cv2.putText(shadow, gesture, (tx + offset, ty + offset),
                    font, font_scale, (0, 0, 0), thickness + 2)
        cv2.addWeighted(shadow, alpha_mul, frame, 1 - alpha_mul, 0, frame)

    glow = frame.copy()
    cv2.putText(glow, gesture, (tx, ty), font, font_scale, color, thickness + 4)
    cv2.addWeighted(glow, 0.4, frame, 0.6, 0, frame)

    cv2.putText(frame, gesture, (tx, ty), font, font_scale, (255, 255, 255), thickness)

    line_y = ty - text_size[1] // 2
    left_end = tx - 20
    right_start = tx + text_size[0] + 20
    if left_end > 10:
        cv2.line(frame, (10, line_y), (left_end, line_y), color, 1)
        cv2.circle(frame, (10, line_y), 3, color, -1)
    if right_start < w - 10:
        cv2.line(frame, (right_start, line_y), (w - 10, line_y), color, 1)
        cv2.circle(frame, (w - 10, line_y), 3, color, -1)

    sub_scale = 0.6 * scale
    sub_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, sub_scale, 1)[0]
    sx = (w - sub_size[0]) // 2
    sy = h - banner_h + 65
    cv2.putText(frame, subtitle, (sx, sy),
                cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (200, 200, 200), 1)

    _draw_corner_badge(frame, gesture, color)


def _draw_corner_badge(frame, gesture, color):
    h, w = frame.shape[:2]
    bx, by = w - 70, 20
    bs = 50

    badge = frame.copy()
    pts = np.array([[bx, by], [bx + bs, by],
                    [bx + bs, by + bs], [bx, by + bs]], dtype=np.int32)
    cv2.fillPoly(badge, [pts], (0, 0, 0))
    cv2.addWeighted(badge, 0.65, frame, 0.35, 0, frame)

    cv2.polylines(frame, [pts], True, color, 2)

    initial = gesture[0]
    ts = cv2.getTextSize(initial, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)[0]
    ix = bx + (bs - ts[0]) // 2
    iy = by + (bs + ts[1]) // 2
    cv2.putText(frame, initial, (ix, iy), cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2)


def draw_fps(frame, fps):
    label = f"FPS  {fps:.0f}"
    ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (18 + ts[0], 26 + ts[1]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, label, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1)


def draw_idle_hint(frame):
    h, w = frame.shape[:2]

    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (101, 101), 0)
    factor = 0.75 + 0.25 * mask
    for c in range(3):
        frame[:, :, c] = np.clip(
            frame[:, :, c].astype(np.float32) * factor, 0, 255
        ).astype(np.uint8)

    hint = "Show a gesture to activate cursed energy"
    font_scale = 0.5
    size = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
    x = (w - size[0]) // 2
    y = h - 18

    pulse = abs(math.sin(time.time() * 1.5))
    gray_val = int(100 + 80 * pulse)
    cv2.putText(frame, hint, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (gray_val, gray_val, gray_val), 1)

    title = "JJK GESTURE SYSTEM"
    ts = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]
    tx = (w - ts[0]) // 2
    cv2.putText(frame, title, (tx, 30), cv2.FONT_HERSHEY_DUPLEX,
                0.6, (60, 60, 60), 1)