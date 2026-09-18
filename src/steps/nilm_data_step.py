from .nilm_common import NilmStep


class NilmDataStep(NilmStep):
    step_type = 'nilm_data'

    def run(self, context):
        if context['manifest'].data.get('steps'):
            raise ValueError('nilm_data must freeze splits before any discovery step; use a new run-id')
        out = self.fresh_dir(context)
        self.worker(context, 'data', out)
        return self.register(context, out, {'split_manifest': 'split_manifest.json', 'data_qc': 'data_qc.json'})


build = NilmDataStep
