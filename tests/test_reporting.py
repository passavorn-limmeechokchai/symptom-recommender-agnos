from concurrent.futures import ThreadPoolExecutor

import pytest

from symptom_rec.reporting import render_section, write_report_section


def test_updating_one_section_preserves_manual_text_and_other_sections(tmp_path):
    path = tmp_path / "REPORT.md"
    path.write_text("# Report\n\nManual introduction.\n\n" + render_section("evaluation", "old") +
                    "\n\nManual middle.\n\n" + render_section("semantic-evaluation", "keep this") + "\n")
    write_report_section("evaluation", "# Generated title\n\n## Results\n\nnew", path)
    content = path.read_text()
    assert "Manual introduction." in content and "Manual middle." in content
    assert render_section("semantic-evaluation", "keep this") in content
    assert "### Results\n\nnew" in content and "old" not in content
    before = path.read_bytes()
    write_report_section("evaluation", "# Generated title\n\n## Results\n\nnew", path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("broken", [
    "<!-- report:evaluation:start -->", "<!-- report:evaluation:end -->",
    "<!-- report:evaluation:end --><!-- report:evaluation:start -->",
    "<!-- report:evaluation:start --><!-- report:evaluation:end -->" * 2,
])
def test_malformed_markers_do_not_overwrite_report(tmp_path, broken):
    path = tmp_path / "REPORT.md"
    path.write_text("Keep the original report.\n" + broken)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        write_report_section("evaluation", "new content", path)
    assert path.read_bytes() == before


def test_parallel_script_sections_are_both_retained(tmp_path):
    path = tmp_path / "REPORT.md"
    path.write_text("# Report\n\nManual notes.\n")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(write_report_section, name, name + " result", path)
                   for name in ["evaluation", "retrieval-evaluation"]]
        for future in futures:
            future.result()
    content = path.read_text()
    assert "Manual notes." in content
    for name in ["evaluation", "retrieval-evaluation"]:
        assert content.count(f"<!-- report:{name}:start -->") == 1
        assert name + " result" in content
