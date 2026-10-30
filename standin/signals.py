"""Turn raw camera/voice measurements from the browser into what the people in
the room would notice.

Code owns the thresholds, so a character only reacts to something that really
happened, and only to observable behavior (never guessed emotions).
"""

from pydantic import BaseModel


class Signals(BaseModel):
    hesitation_s: float | None = None  # silence between their last word and the user's first
    speak_s: float | None = None  # how long the user spoke
    syllables_per_s: float | None = None
    fillers: int = 0  # 어, 음, 그..., um, uh
    camera: bool = False
    eye_contact_pct: float | None = None  # share of speaking time looking at the screen
    looking_down_pct: float | None = None
    face_missing_pct: float | None = None


CUES = {
    "en": {"silent": "stayed silent for a long time", "pause": "paused {n}s before answering", "fast": "spoke very fast",
           "slow": "spoke slowly, haltingly", "fillers": "said 'uh/um' {n} times", "away": "was mostly out of view",
           "down": "looked down for {n}% of the answer", "avoid": "kept looking away (eye contact {n}%)", "steady": "held steady eye contact"},
    "ko": {"silent": "한참 동안 말이 없었음", "pause": "답하기 전에 {n}초 머뭇거림", "fast": "말이 매우 빨랐음",
           "slow": "말이 느리고 자주 끊겼음", "fillers": "'어/음' 같은 추임새 {n}번", "away": "대부분 화면 밖에 있었음",
           "down": "답하는 시간의 {n}% 동안 아래를 봄", "avoid": "시선을 자주 피함 (눈 맞춤 {n}%)", "steady": "눈을 계속 맞춤"},
    "ja": {"silent": "長い間黙っていた", "pause": "答える前に{n}秒ためらった", "fast": "とても早口だった",
           "slow": "ゆっくりで途切れがちだった", "fillers": "「えー」が{n}回", "away": "ほとんど画面の外にいた",
           "down": "回答の{n}%で下を向いていた", "avoid": "視線をよくそらした(アイコンタクト{n}%)", "steady": "視線をしっかり合わせた"},
}


def describe(sig: Signals, ui: str = "en") -> str:
    """Only what stands out; empty when nothing would be noticed."""
    say = CUES.get(ui, CUES["en"])
    notes = []
    if sig.hesitation_s is not None and sig.hesitation_s >= 3:
        notes.append(say["silent"] if sig.hesitation_s >= 30 else say["pause"].format(n=round(sig.hesitation_s)))
    if sig.syllables_per_s is not None and (sig.speak_s or 0) >= 3:
        if sig.syllables_per_s >= 8:
            notes.append(say["fast"])
        elif sig.syllables_per_s <= 3:
            notes.append(say["slow"])
    if sig.fillers >= 3:
        notes.append(say["fillers"].format(n=sig.fillers))
    if sig.camera:
        if (sig.face_missing_pct or 0) >= 40:
            notes.append(say["away"])
        elif (sig.looking_down_pct or 0) >= 40:
            notes.append(say["down"].format(n=round(sig.looking_down_pct)))
        elif sig.eye_contact_pct is not None and sig.eye_contact_pct < 50:
            notes.append(say["avoid"].format(n=round(sig.eye_contact_pct)))
        elif sig.eye_contact_pct is not None and sig.eye_contact_pct >= 85 and (sig.speak_s or 0) >= 4:
            notes.append(say["steady"])
    return "; ".join(notes)
