import os

import cv2
import mediapipe as mp
import numpy as np

SOURCE = 0                 
MIRROR = True              
RECORD = True              
OUT_W, OUT_H = 720, 1280   
HALF_H = OUT_H // 2
CAT_DIR = "cats"
CATS = {
    "pointing": "pointing.jpg",   # 👉🏻👈🏻
    "mouth": "mouth.jpg",         # hand on mouth
    "salute": "salute.jpg",       #   "Yes sir"
    "shaka": "shaka.jpg",         # 🤙🏻
}
CONFIRM_FRAMES = 2         
RELEASE_FRAMES = 8         

CUT_MARGIN = 0.03

PREVIEW_W = 640            
SHOW_W, SHOW_H = 340, 604  
DETECT_W = 640             

MOUTH_DIST = 0.45          
SALUTE_DIST = 0.60         

TOP_ZONE = 0.30

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh


def cover_resize(img, w, h):
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    img = cv2.resize(img, (int(iw * s) + 1, int(ih * s) + 1))
    y0 = (img.shape[0] - h) // 2
    x0 = (img.shape[1] - w) // 2
    return img[y0:y0 + h, x0:x0 + w]


def to_px(lms, w, h):
    return np.array([[l.x * w, l.y * h] for l in lms.landmark], dtype=np.float32)


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def fingers(p):
    
    size = dist(p[0], p[9])
    thumb = dist(p[4], p[5]) > 0.7 * size and dist(p[4], p[17]) > dist(p[3], p[17])
    others = [dist(p[0], p[t]) > dist(p[0], p[pip]) * 1.05
              for t, pip in ((8, 6), (12, 10), (16, 14), (20, 18))]
    return (thumb, *others)


def is_shaka(p):
    t, i, m, r, pk = fingers(p)
    return t and pk and not (i or m or r)


def is_point(p):
    t, i, m, r, pk = fingers(p)
    return i and not (m or r or pk)


def is_open(p):
    t, i, m, r, pk = fingers(p)
    return sum((i, m, r, pk)) >= 3


def is_sideways(p):
    return abs(p[12][1] - p[0][1]) < 0.6 * dist(p[0], p[12])


def detect(hands, face, frame_h, allow_fallback):
     
    for p in hands:
        if is_shaka(p):
            return "shaka"

    if len(hands) == 2 and all(is_point(p) for p in hands):
        size = (dist(hands[0][0], hands[0][9]) + dist(hands[1][0], hands[1][9])) / 2
        if dist(hands[0][8], hands[1][8]) < 1.4 * size:
            return "pointing"

    if face is not None:
        fw = dist(face[234], face[454])
        mouth = (face[13] + face[14]) / 2
        nose_y = face[1][1]

        for p in hands:
            for k in (4, 8, 12, 16, 20, 5, 9, 13, 17):
                q = p[k]
                if q[1] > nose_y and dist(q, mouth) < MOUTH_DIST * fw:
                    return "mouth"

        head_ids = [10, 21, 251, 105, 334, 54, 284, 67, 297, 103, 332]
        for p in hands:
            t, i, m, r, pk = fingers(p)
            if sum((i, m, r, pk)) < 2:
                continue
            tips = [p[k] for k in (8, 12, 16, 20)]
            if min(q[1] for q in tips) >= nose_y:
                continue
            near = min(dist(q, face[k]) for q in tips for k in head_ids)
            if near < SALUTE_DIST * fw:
                return "salute"
        return None

    if allow_fallback:
        for p in hands:
            if p[:, 1].min() < TOP_ZONE * frame_h:
                if is_open(p) and is_sideways(p):
                    return "salute"
                return "mouth"
    return None


class HandCam:
    def __init__(self, w, h):
        self.W, self.H = w, h
        self.aspect = OUT_W / HALF_H
        self.cx, self.cy, self.cw = w / 2, h / 2, w * 0.6

    def update(self, hand_pts):
        if hand_pts:
            allp = np.vstack(hand_pts)
            x0, y0 = allp.min(axis=0)
            x1, y1 = allp.max(axis=0)
            bw, bh = (x1 - x0), (y1 - y0)
            tw = max(bw * 1.8, bh * 1.8 * self.aspect, self.W * 0.35)
            tcx, tcy = (x0 + x1) / 2, (y0 + y1) / 2
            a = 0.2
            self.cx += (tcx - self.cx) * a
            self.cy += (tcy - self.cy) * a
            self.cw += (tw - self.cw) * a

    def rect(self):
        cw = int(min(self.cw, self.W, self.H * self.aspect))
        ch = int(cw / self.aspect)
        x0 = int(np.clip(self.cx - cw / 2, 0, self.W - cw))
        y0 = int(np.clip(self.cy - ch / 2, 0, self.H - ch))
        return x0, y0, x0 + cw, y0 + ch


def main():
    cats = {}
    for k, f in CATS.items():
        img = cv2.imread(os.path.join(CAT_DIR, f))
        if img is None:
            raise SystemExit(f"Missing image: {os.path.join(CAT_DIR, f)}")
        cats[k] = cover_resize(img, OUT_W, HALF_H)
    blank = np.zeros((HALF_H, OUT_W, 3), np.uint8)

    cap = cv2.VideoCapture(SOURCE)
    if SOURCE == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6,
                           min_tracking_confidence=0.5)
    face = mp_face.FaceMesh(max_num_faces=1, min_detection_confidence=0.4,
                            min_tracking_confidence=0.4)

    cv2.namedWindow("Me", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Reel", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("Me", 20, 40)
    cv2.moveWindow("Reel", PREVIEW_W + 60, 20)

    writer = None
    cam = None
    active, pending, count = None, None, 0
    last_face, face_lost = None, 99
    cut_y = None                      

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if MIRROR:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        if cam is None:
            cam = HandCam(w, h)

        dh = int(h * DETECT_W / w)
        det = cv2.resize(frame, (DETECT_W, dh))
        rgb = cv2.cvtColor(det, cv2.COLOR_BGR2RGB)
        hr = hands.process(rgb)
        fr = face.process(rgb)
        hand_pts = [to_px(l, w, h) for l in (hr.multi_hand_landmarks or [])]
        face_pts = to_px(fr.multi_face_landmarks[0], w, h) if fr.multi_face_landmarks else None

        if face_pts is not None:
            last_face, face_lost = face_pts, 0
        else:
            face_lost += 1
        face_for_det = face_pts if face_pts is not None else (last_face if face_lost < 10 else None)

        if face_pts is not None:
            fw = dist(face_pts[234], face_pts[454])
            target = face_pts[0][1] - CUT_MARGIN * fw
            cut_y = target if cut_y is None else cut_y + (target - cut_y) * 0.3

        cand = detect(hand_pts, face_for_det, h, allow_fallback=(cut_y is None))
        if cand == pending:
            count += 1
        else:
            pending, count = cand, 1
        need = CONFIRM_FRAMES if pending else RELEASE_FRAMES
        if count >= need and active != pending:
            active = pending
            print("gesture ->", active)

        cy0 = 0 if cut_y is None else int(np.clip(cut_y, 0, h - 120))
        view = frame[cy0:]
        vh = view.shape[0]

        shifted = [p - np.array([0, cy0], np.float32) for p in hand_pts]
        cam.H = vh
        cam.update(shifted)
        x0, y0, x1, y1 = cam.rect()
        bottom = cv2.resize(view[y0:y1, x0:x1], (OUT_W, HALF_H))

        top = cats[active] if active else blank
        out = np.vstack([top, bottom])

        if RECORD:
            if writer is None:
                writer = cv2.VideoWriter("reel.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                                         30, (OUT_W, OUT_H))
            writer.write(out)

        cv2.imshow("Me", cv2.resize(view, (PREVIEW_W, max(1, int(vh * PREVIEW_W / w)))))
        cv2.imshow("Reel", cv2.resize(out, (SHOW_W, SHOW_H)))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

