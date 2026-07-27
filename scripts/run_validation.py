from pathlib import Path

from smc_validator.validation.prev_day_low_retest import format_report, run_validation

REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "prev_day_low_validation.md"


def main() -> None:
    results = run_validation()
    report = format_report(results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
