from .nilm_common import NilmStep, strict_guard


class NilmSelectStep(NilmStep):
    step_type = 'nilm_select'

    def run(self, context):
        strict_guard(context)
        out = self.fresh_dir(context)
        self.worker(context, 'select', out)
        return self.register(context, out, {'selection': 'selection.json'})


build = NilmSelectStep
