"""FileKillSwitch.engaged() must fail closed: an unreadable/corrupt state
file must be treated as engaged (pause trading), not as "not engaged"
(continue trading). Silently defeating the emergency stop on a file read
error is the opposite of what a kill switch is for.
"""
from trader.watchdog import FileKillSwitch


def test_engaged_is_false_when_file_does_not_exist(tmp_path):
    switch = FileKillSwitch(tmp_path / "kill.json")
    assert switch.engaged() is False


def test_engaged_reflects_true_content(tmp_path):
    path = tmp_path / "kill.json"
    switch = FileKillSwitch(path)
    switch.engage("test halt")
    assert switch.engaged() is True


def test_engaged_reflects_false_content_after_disengage(tmp_path):
    path = tmp_path / "kill.json"
    switch = FileKillSwitch(path)
    switch.engage("test halt")
    switch.disengage()
    assert switch.engaged() is False


def test_engaged_fails_closed_on_corrupt_file(tmp_path):
    path = tmp_path / "kill.json"
    path.write_text("{not valid json", encoding="utf-8")
    switch = FileKillSwitch(path)
    assert switch.engaged() is True


def test_engaged_fails_closed_on_empty_file(tmp_path):
    path = tmp_path / "kill.json"
    path.write_text("", encoding="utf-8")
    switch = FileKillSwitch(path)
    assert switch.engaged() is True
