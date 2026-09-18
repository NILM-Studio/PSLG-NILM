from .nilm_common import NilmStep, strict_guard


class NilmEvaluateStep(NilmStep):
    step_type = 'nilm_evaluate'

    def run(self, context):
        strict_guard(context)
        out = self.fresh_dir(context)
        self.worker(context, 'evaluate', out)
        return self.register(context, out, {'results': 'results.json', 'predictions': 'prediction_manifest.json'})


build = NilmEvaluateStep
