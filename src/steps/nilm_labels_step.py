from .nilm_common import NilmStep, strict_guard


class NilmLabelsStep(NilmStep):
    step_type = 'nilm_labels'

    def run(self, context):
        strict_guard(context)
        out = self.fresh_dir(context)
        self.worker(context, 'labels', out)
        return self.register(context, out, {'metadata': 'metadata.json', 'quality': 'label_qc.json'})


build = NilmLabelsStep
