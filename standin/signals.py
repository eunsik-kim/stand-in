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


def describe(sig: Signals) -> str:
    """Only what stands out; empty when nothing would be noticed."""
    notes = []
    if sig.hesitation_s is not None and sig.hesitation_s >= 3:
        notes.append(f"paused {sig.hesitation_s:.0f}s before answering")
    if sig.syllables_per_s is not None and (sig.speak_s or 0) >= 3:
        if sig.syllables_per_s >= 8:
            notes.append("spoke very fast")
        elif sig.syllables_per_s <= 3:
            notes.append("spoke slowly, haltingly")
    if sig.fillers >= 3:
        notes.append(f"said 'uh/um' {sig.fillers} times")
    if sig.camera:
        if (sig.face_missing_pct or 0) >= 40:
            notes.append("was mostly out of view")
        elif (sig.looking_down_pct or 0) >= 40:
            notes.append(f"looked down (at notes?) for {sig.looking_down_pct:.0f}% of the answer")
        elif sig.eye_contact_pct is not None and sig.eye_contact_pct < 50:
            notes.append(f"kept looking away (eye contact {sig.eye_contact_pct:.0f}% of the time)")
        elif sig.eye_contact_pct is not None and sig.eye_contact_pct >= 85 and (sig.speak_s or 0) >= 4:
            notes.append("held steady eye contact")
    return "; ".join(notes)
