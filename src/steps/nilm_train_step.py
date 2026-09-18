from .nilm_common import NilmStep, strict_guard


class NilmTrainStep(NilmStep):
    step_type = 'nilm_train'

    def run(self, context):
        strict_guard(context)
        # Worker supports resuming completed immutable trials, never overwrites partial runs.
        from pathlib import Path
        out = Path(self.log_dir(context))
        self.worker(context, 'train', out)
        return self.register(context, out, {'trials': 'trials.json'})


build = NilmTrainStep
