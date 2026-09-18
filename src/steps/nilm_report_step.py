from pathlib import Path
from .nilm_common import NilmStep, read_json, write_json


class NilmReportStep(NilmStep):
    step_type = 'nilm_report'

    def run(self, context):
        out = Path(self.log_dir(context))
        self.worker(context, 'report', out)
        report = read_json(out / 'report.json')
        if report['test'] is not None:
            from src.utils.nilm_report_summary import summarize_test
            summary = summarize_test(report['test']['results'], context['appliance'])
            report.update(summary)
            write_json(out / 'report.json', report)
            with (out / 'report.md').open('a', encoding='utf-8') as f:
                f.write('\n## Paired held-out household results\n\n')
                f.write('| Model | Comparison | Houses | MAE improvement (W) | 95% household CI |\n|---|---|---:|---:|---|\n')
                for s in summary['paired_test']:
                    f.write(f"| {s['model']} | {s['baseline']} minus {s['proposed']} | {s['households']} | {s['mae_improvement_w']:.4f} | {s['ci95']} |\n")
                f.write('\n## Held-out power and activity metrics\n\n')
                f.write('| House | Model | Case | MAE mean (W) | Seed SD | Active MAE | Inactive MAE | Activity recall |\n|---|---|---|---:|---:|---:|---:|---:|\n')
                def display(value):
                    return 'NA' if value is None else f'{value:.4f}'
                for s in summary['detailed_metrics']['household_means']:
                    f.write(f"| {s['house']} | {s['model']} | {s['case']} | {display(s['mae']['mean'])} | {display(s['mae']['seed_std'])} | {display(s['active_MAE']['mean'])} | {display(s['inactive_MAE']['mean'])} | {display(s['activity_recall']['mean'])} |\n")
        return self.register(context, out, {'report': 'report.md', 'results': 'report.json'})


build = NilmReportStep
