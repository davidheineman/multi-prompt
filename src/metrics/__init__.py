import requests, os, json


PRETTY_METRIC_NAMES = {
    "bleu": "BLEU",
    "sari": "SARI",
    "bert_score": "BERTScore",
    "lens": "LENS",
    "lens_salsa": "LENS-SALSA",
    "sle": "SLE",
    "comet": "COMET",
    "comet_20": "COMET",
    "comet_22": "COMET",
    "comet_kiwi": "COMET Kiwi",
    "comet_kiwi_22": "COMET Kiwi",
    "comet_kiwi_23": "COMET Kiwi",
    "comet_kiwi_xxl": "COMET Kiwi XXL",
    "xcomet": "XCOMET",
    "metricx": "MetricX ↓",
    "metricx_qe": "MetricX QE ↓",
    "unite": "UniTE",
    "bart_score": "BARTScore",
    "code_bert_score": "CodeBERTScore",
    "pass@1": "pass@1",
    "pass@k": "pass@k",
    "unique_bigrams": "Unique Bigrams",
    "candidate_agreement": "Candidate Agreement",
    "candidate_quality": "Candidate Quality"
}

METRIC_PORT = int(os.environ.get("METRIC_PORT", 8501))


class AbstractMetric:
    def __init__(self, name, **kwargs):
        self.name = name
        self.devices = kwargs.get('devices', None)
        self.batch_size = kwargs.get('batch_size', 1)
        self.requires_references = kwargs.get('requires_references', False)
        
        if not kwargs.get('is_endpoint', False):
            print(f'Initalizing metric {self}...')

    def __call__(self, src, pred, ref):
        """
        All metric functions are in the form of (source, prediction, reference), although not all three
        fields may be used. 

        src:    Array of source texts
        pred:   Array of predicted texts
        ref:    Array of arrays of reference texts for each source text

        e.g.,
            sources=["About 95 species are currently accepted."]
            predictions=["About 95 you now get in."]
            references=[["About 95 species are currently known.", "About 95 species are now accepted.", "95 species are now accepted."]]

        Returns: System-level scores across an entire dataset
        """
        raise NotImplementedError()
    
    def __str__(self):
        return PRETTY_METRIC_NAMES.get(self.name, self.name)
    
    def __repr__(self):
        return f'{self.__str__()}()'
    
    def get_endpoint_name(self):
        return 'http://localhost:{port}/' + self.name + '_eval'
    

class MetricEndpoint(AbstractMetric):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, is_endpoint=True, **kwargs)
        self.name = name
        self.requires_references = kwargs.get('ref', False)

    def __call__(self, src=None, pred=None, ref=None):
        endpoint = self.get_endpoint_name()

        sent = {}
        if src: sent['src'] = src
        if pred: sent['pred'] = pred
        if ref: sent['ref'] = ref

        try:
            response = requests.post(endpoint.format(port=METRIC_PORT), json=sent)
            response = response.json()
        except json.decoder.JSONDecodeError:
            raise RuntimeError(f"{self.name} endpoint failed to respond. Returned: {response}")
        scores = response['scores']
        
        return scores
