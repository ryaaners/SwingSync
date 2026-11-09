"""Describe agreement with independently supplied manual points; never admit coaching."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def evaluate(*, run, run_sha256, video, reference, output):
    from backend.swingsync.movement.pose_artifacts import (
        ROOT, _json, assert_artifact_unchanged, load_pose_artifact, sha256,
    )
    from backend.swingsync.pose_runtime.extraction import publish_json
    from backend.swingsync.research.pose_validation import PROTOCOL_PATH, evaluate_reference, load_protocol

    run, video, reference, output = map(Path, (run, video, reference, output))
    if output.exists():
        raise ValueError('Output exists; refusing to overwrite agreement evidence')
    artifact = load_pose_artifact(run, expected_run_sha256=run_sha256, video_path=video)
    sources = [Path(__file__).resolve(), ROOT / 'backend/swingsync/research/pose_validation.py',
               ROOT / 'backend/swingsync/movement/pose_artifacts.py']
    identity = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    reference_sha, protocol_sha = sha256(reference), sha256(PROTOCOL_PATH)
    report = evaluate_reference(_json(reference, reference_sha), artifact.result,
                                run_sha256=artifact.run_sha256, result_sha256=artifact.result_sha256,
                                protocol=load_protocol())
    assert_artifact_unchanged(artifact)
    if (sha256(reference) != reference_sha or sha256(PROTOCOL_PATH) != protocol_sha
            or {str(p.relative_to(ROOT)): sha256(p) for p in sources} != identity):
        raise ValueError('Reference/protocol/evaluation source drift; no agreement report published')
    report.update(reference_file_sha256=reference_sha, protocol_file_sha256=protocol_sha,
                  evaluation_source_sha256=identity, created_at=datetime.now(timezone.utc).isoformat(),
                  empirical_accuracy_study_completed=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    publish_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('run', 'video', 'reference', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--run-sha256', required=True)
    args = parser.parse_args()
    try:
        report = evaluate(**vars(args))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'Reference agreement unavailable: {exc}\n')
    print(f"Saved conditional reference agreement: {report['scored_count']} points, "
          f"{report['model_abstention_count']} model abstentions; coaching remains unavailable.")


if __name__ == '__main__':
    main()
