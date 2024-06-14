import torch, transformers
from datasets import Dataset

from sacrebleu.metrics import BLEU as ScareBLEU
from bert_score import score as _bert_score
from comet import download_model, load_from_checkpoint

from .metricx23.models import MT5ForRegression
from . import AbstractMetric


class BLEU(AbstractMetric):
    def __init__(self, **kwargs):
        super().__init__(name='bleu', requires_references=True, **kwargs)

    def __call__(self, pred, ref, src=None):
        bleu_metric = ScareBLEU(effective_order=True)
        evaluation = []
        for p, r in zip(pred, ref):
            evaluation += [bleu_metric.sentence_score(p, r).score]
        return evaluation


class BertScore(AbstractMetric):
    def __init__(self, **kwargs):
        super().__init__(name='bert_score', requires_references=True, **kwargs)

    def __call__(self, pred, ref, src=None):
        _, _, F1 = _bert_score(pred, ref, lang="en", verbose=False)
        return [float(f1)*100 for f1 in F1]


class BartScore(AbstractMetric):
    def __init__(self, batch_size=4, **kwargs):
        from .bart_score.bart_score import BARTScorer
        super().__init__(name='bart_score', batch_size=batch_size, requires_references=True, **kwargs)

        self.metric = BARTScorer(checkpoint='facebook/bart-large-cnn', device='cuda')
        self.metric.load(path='bart_score/bart-checkpoint.pth')

    def __call__(self, pred, ref, src=None):
        ref_lengths = set(len(l) for l in ref)
        if len(ref_lengths) > 1:
            print('Found unequal reference lengths, batching BARTScores by reference length...')
            evaluation = []
            batches = [[(r, p) for r, p in zip(ref, pred) if len(r) == l] for l in ref_lengths]
            for batch in batches:
                ref, pred = [i[0] for i in batch], [i[1] for i in batch]
                evaluation += self.metric.multi_ref_score(pred, ref, agg="max", batch_size=self.batch_size)
        else:
            evaluation = self.metric.multi_ref_score(pred, ref, agg="max", batch_size=self.batch_size)
        return evaluation


class Comet(AbstractMetric):
    """
    See https://huggingface.co/Unbabel
    """
    def __init__(self, variation='comet_22', batch_size=4, size=None, **kwargs):
        super().__init__(name=variation, batch_size=batch_size, requires_references=('kiwi' in variation), **kwargs)

        match self.name:
            case 'comet_20': model_id = "Unbabel/wmt20-comet-da"
            case 'comet_22': model_id = "Unbabel/wmt22-comet-da"
            case 'comet_unite_22': model_id = "Unbabel/wmt22-unite-da" 
            case 'comet_kiwi_22': model_id = "Unbabel/wmt22-cometkiwi-da"
            case 'comet_kiwi_23':
                match size:
                    case 'xl': model_id = "Unbabel/wmt23-cometkiwi-da-xl"
                    case 'xxl': model_id = "Unbabel/wmt23-cometkiwi-da-xxl"
                    case _: raise NotImplementedError(f'COMET Kiwi 23 size {size} not supported!')
            case _: raise NotImplementedError(f'COMET variation {self.name} not supported!')
        
        model_path = download_model(model_id)
        self.metric = load_from_checkpoint(model_path)

    def _comet(self, src, pred, ref):
        assert 'kiwi' not in self.name
        data = []
        for s, p, r in zip(src, pred, ref):
            if isinstance(r, list): r = r[0]
            data += [{ 'src': s, 'mt': p, 'ref': r }]
        evaluation = self.metric.predict(data, batch_size=self.batch_size, devices=self.devices, gpus=len(self.devices))
        return [s*100 for s in evaluation.scores]

    def _comet_kiwi(self, src, pred, ref=None):
        assert 'kiwi' in self.name
        data = []
        for s, p in zip(src, pred):
            data += [{ 'src': s, 'mt': p}]
        evaluation = self.metric.predict(data, batch_size=self.batch_size, devices=self.devices, gpus=len(self.devices))
        return [s*100 for s in evaluation.scores]
    
    def __call__(self, src, pred, ref=None):
        if 'kiwi' in self.name:
            return self._comet_kiwi(src=src, pred=pred, ref=None)
        else:
            assert ref is not None, 'COMET requires references!'
            return self._comet(src=src, pred=pred, ref=ref)
        

class XComet(AbstractMetric):
    def __init__(self, size='xl', batch_size=2, **kwargs):
        super().__init__(name='xcomet', batch_size=batch_size, requires_references=True, **kwargs)

        match size:
            case 'xl': model_id = "Unbabel/XCOMET-XL"
            case 'xxl': model_id = "Unbabel/XCOMET-XXL"
            case _: raise NotImplementedError(f'XCOMET size {size} not supported!')

        model_path = download_model(model_id)
        self.metric = load_from_checkpoint(model_path)

    def __call__(self, src, pred, ref):
        data = []
        for s, p, r in zip(src, pred, ref):
            if isinstance(r, list): r = r[0]
            data += [{ 'src': s, 'mt': p, 'ref': r }]
        evaluation = self.metric.predict(data, batch_size=self.batch_size, devices=self.devices, gpus=len(self.devices))
        return [s*100 for s in evaluation.scores]


class MetricX(AbstractMetric):
    def __init__(self, variation='metricx', size='xl', batch_size=8, **kwargs):
        super().__init__(name=variation, batch_size=batch_size, requires_references=('qe' in variation), **kwargs)

        is_qe = (self.name == 'metricx_qe')

        model_id = None
        if is_qe:
            match size:
                case 'l': model_id = "google/metricx-23-qe-l-v2p0"
                case 'xl': model_id = "google/metricx-23-qe-xl-v2p0"
                case 'xxl': model_id = "google/metricx-23-qe-xxl-v2p0"
        else:
            match size:
                case 'l': model_id = "google/metricx-23-l-v2p0"
                case 'xl': model_id = "google/metricx-23-xl-v2p0"
                case 'xxl': model_id = "google/metricx-23-xxl-v2p0"
        if model_id is None: raise NotImplementedError(f'MetricX variation {self.name} at size {size} not supported!')

        device = torch.device("cuda")

        per_device_batch_size = self.batch_size # batch_size // (len(devices) if isinstance(devices, list) else devices)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained("google/mt5-xl")

        model = MT5ForRegression.from_pretrained(model_id)
        model.to(device)
        model.eval()

        training_args = transformers.TrainingArguments(
            output_dir='/dev',
            per_device_eval_batch_size=per_device_batch_size,
            dataloader_pin_memory=False,
        )
        self.metric = transformers.Trainer(model=model, args=training_args)

    def prepare_inputs(self, data, is_qe, max_input_length=1024):
        """
        Custom function for creating a MetricX dataset class using a lists for input.
        """
        def _make_input(example):
            if is_qe:
                example["input"] = f'candidate: {example["hypothesis"]} source: {example["source"]}'
            else:
                example["input"] = f'candidate: {example["hypothesis"]} reference: {example["reference"]}'
            return example

        def _tokenize(example):
            tokenized = self.tokenizer(
                example["input"],
                max_length=max_input_length,
                truncation=True,
                padding='max_length',
            )
            return tokenized

        def _remove_eos(example):
            example["input_ids"] = example["input_ids"][:-1]
            example["attention_mask"] = example["attention_mask"][:-1]
            return example

        ds = Dataset.from_list(data)
        ds = ds.map(_make_input)
        ds = ds.map(_tokenize)
        ds = ds.map(_remove_eos)
        ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"],
            device=self.metric.model.device,
            output_all_columns=True,
        )
        return ds
    
    def _metricx(self, src, pred, ref):
        data = []
        for s, p, r in zip(src, pred, ref):
            if isinstance(r, list):
                r = r[0]
            data.append({'source': s, 'hypothesis': p, 'reference': r})
          
        ds = self.prepare_inputs(data, is_qe=False)
        evaluation, _, _ = self.metric.predict(test_dataset=ds)

        # Very important! Lower is better for MetricX, so we negate before returning. Note the 
        # model evaluation logic does not do this when reporting final results.
        evaluation = -evaluation

        return evaluation.tolist()

    def _metricx_qe(self, src, pred, ref=None):
        data = []
        for s, p in zip(src, pred):
            data += [{ 'source': s, 'hypothesis': p }]

        ds = self.prepare_inputs(data, is_qe=True)
        evaluation, _, _ = self.metric.predict(test_dataset=ds)

        # Very important! Lower is better for MetricX, so we negate before returning. Note the 
        # model evaluation logic does not do this when reporting final results.
        evaluation = -evaluation

        return evaluation.tolist()
    
    def __call__(self, src, pred, ref=None):
        if 'qe' in self.name:
            return self._metricx_qe(src=src, pred=pred, ref=None)
        else:
            assert ref is not None, 'MetricX requires references!'
            return self._metricx(src=src, pred=pred, ref=ref)
