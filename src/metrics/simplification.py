from collections import Counter

from sacrebleu.metrics import bleu as ScareBLEU
from lens import LENS, LENS_SALSA, download_model

from . import AbstractMetric
from .sle.sle.scorer import SLEScorer

# Quiet PyTorch Lightning logger
import logging
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)


class Lens(AbstractMetric):
    """ https://aclanthology.org/2023.acl-long.905 """
    def __init__(self, variation='lens', batch_size=16, **kwargs):
        super().__init__(name=variation, batch_size=batch_size, requires_references=(variation == 'lens'), **kwargs)

        match self.name:
            case 'lens': 
                model_path = download_model("davidheineman/lens")
                self.metric = LENS(model_path, rescale=True)
            case 'lens_salsa': 
                model_path = download_model("davidheineman/lens-salsa")
                self.metric = LENS_SALSA(model_path)
            case _: raise NotImplementedError(f'LENS variation {self.name} not supported!')

    def _lens(self, src, pred, ref):
        evaluation = self.metric.score(src, pred, ref, batch_size=self.batch_size, devices=self.devices)
        return evaluation

    def _lens_salsa(self, src, pred, ref=None):
        evaluation, _ = self.metric.score(src, pred, batch_size=self.batch_size, devices=self.devices)
        return evaluation
    
    def __call__(self, src, pred, ref=None):
        match self.name:
            case 'lens_salsa': 
                return self._lens_salsa(src=src, pred=pred, ref=None)
            case 'lens': 
                assert ref is not None, 'LENS requires references!'
                return self._lens(src=src, pred=pred, ref=ref)
            case _: raise NotImplementedError(f'LENS variation {self.name} not supported!')
        

class SLE(AbstractMetric):
    """ https://aclanthology.org/2023.emnlp-main.739 """
    def __init__(self, **kwargs):
        super().__init__(name='sle', requires_references=False, **kwargs)

        self.metric = SLEScorer("liamcripwell/sle-base")

    def __call__(self, src, pred, ref=None):
        evaluation = self.metric.score(pred, inputs=src)
        return evaluation['sle_delta']
        

class Sari(AbstractMetric):
    """
    SARI - https://aclanthology.org/Q16-1029
    
    Adapted from:
    https://github.com/huggingface/evaluate/blob/main/metrics/sari/sari.py
    Based on:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/sari_hook.py
    Which was originally based on:
    https://github.com/cocoxu/simplification/blob/master/SARI.py

    A quick note. The original implementation had an "alleged bug": https://github.com/cocoxu/simplification/issues/6
    where a fix was implemented in the tensor version. This fix changes the computation and has become the dominantely used
    version, although there is not an "official" SARI implementation. Thus, this code has gone through these revisions. 
    PyTorch -> Tensorflow -> PyTorch (within HuggingFace) -> Standalone PyTorch

    I do not use the HF evaluate because it has an incredibly high overhead when computing individual sentences.

    This uses the ScareBLEU tokenizer (v. >=2), which is the predominately used tokenizer in papers evaluating with ASSET.
    """
    def __init__(self, tokenizer="13a", **kwargs):
        super().__init__(name='sari', requires_references=True, **kwargs)
        self.tokenizer = ScareBLEU._get_tokenizer(tokenizer)()

    def __call__(self, src, pred, ref):
        evaluation = []
        for s, p, r in zip(src, pred, ref):
            score = 100 * self.sari_sent(self.normalize(s), self.normalize(p), [self.normalize(sent) for sent in r])
            evaluation += [score]
        return evaluation
    
    def sari_ngram_new(self, sgrams, cgrams, rgramslist, numref):
        """ My attempt at refactoring sari_ngram. This does not work. """
        # Get n-grams in source, complex, and simplified reference sentences
        sgrams = Counter(sgrams)
        cgrams = Counter(cgrams)
        rgrams = Counter(rgram for rgrams in rgramslist for rgram in rgrams)

        # Repeated tokens for source and candidate n-grams
        sgrams_rep = Counter({gram: count * numref for gram, count in sgrams.items()})
        cgrams_rep = Counter({gram: count * numref for gram, count in cgrams.items()})

        # KEEP
        keepgram_rep = sgrams_rep & cgrams_rep
        keepgram_rep_good = keepgram_rep & rgrams
        keepscore = sum(keepgram_rep_good.values())

        p = keepscore / len(keepgram_rep) if keepgram_rep else 1
        r = keepscore / sum(rgrams.values()) if rgrams else 1

        keep = (2 * p * r) / (p + r) if (p > 0 or r > 0) and (keepgram_rep or rgrams) else 0

        # DELETION
        delgram_rep = sgrams_rep - cgrams_rep
        delgram_rep_good = delgram_rep - rgrams
        delscore = sum(delgram_rep_good.values())
        
        del_ = delscore / len(delgram_rep) if delgram_rep else 1 # Deletion score only considers precision

        # ADDITION
        addgram = set(cgrams) - set(sgrams)
        addgram_good = set(addgram) & set(rgrams)
        addscore = len(addgram_good)

        p = addscore / len(addgram) if addgram else 1
        r = addscore / len(rgrams - sgrams) if rgrams - sgrams else 1

        add_ = (2 * p * r) / (p + r) if (p > 0 or r > 0) and (addgram or (rgrams - sgrams)) else 0

        return (keep, del_, add_)

    def sari_ngram(self, sgrams, cgrams, rgramslist, numref):
        rgramsall = [rgram for rgrams in rgramslist for rgram in rgrams]
        rgramcounter = Counter(rgramsall)

        sgramcounter = Counter(sgrams)
        sgramcounter_rep = Counter()
        for sgram, scount in sgramcounter.items():
            sgramcounter_rep[sgram] = scount * numref

        cgramcounter = Counter(cgrams)
        cgramcounter_rep = Counter()
        for cgram, ccount in cgramcounter.items():
            cgramcounter_rep[cgram] = ccount * numref

        # KEEP
        keepgramcounter_rep = sgramcounter_rep & cgramcounter_rep
        keepgramcountergood_rep = keepgramcounter_rep & rgramcounter
        keepgramcounterall_rep = sgramcounter_rep & rgramcounter

        keeptmpscore1 = 0
        keeptmpscore2 = 0
        for keepgram in keepgramcountergood_rep:
            keeptmpscore1 += keepgramcountergood_rep[keepgram] / keepgramcounter_rep[keepgram]
            keeptmpscore2 += keepgramcountergood_rep[keepgram] # Fix an alleged bug [2] in the keep score computation.

        keepscore_precision = 1
        keepscore_recall = 1
        if len(keepgramcounter_rep) > 0:
            keepscore_precision = keeptmpscore1 / len(keepgramcounter_rep)
        if len(keepgramcounterall_rep) > 0:
            keepscore_recall = keeptmpscore2 / sum(keepgramcounterall_rep.values()) # Fix an alleged bug [2] in the keep score computation.
        keepscore = 0
        if keepscore_precision > 0 or keepscore_recall > 0:
            keepscore = 2 * keepscore_precision * keepscore_recall / (keepscore_precision + keepscore_recall)

        # DELETION
        delgramcounter_rep = sgramcounter_rep - cgramcounter_rep
        delgramcountergood_rep = delgramcounter_rep - rgramcounter
        delgramcounterall_rep = sgramcounter_rep - rgramcounter
        deltmpscore1 = 0
        deltmpscore2 = 0
        for delgram in delgramcountergood_rep:
            deltmpscore1 += delgramcountergood_rep[delgram] / delgramcounter_rep[delgram]
            deltmpscore2 += delgramcountergood_rep[delgram] / delgramcounterall_rep[delgram]

        delscore_precision = 1
        if len(delgramcounter_rep) > 0:
            delscore_precision = deltmpscore1 / len(delgramcounter_rep)

        # ADDITION
        addgramcounter = set(cgramcounter) - set(sgramcounter)
        addgramcountergood = set(addgramcounter) & set(rgramcounter)
        addgramcounterall = set(rgramcounter) - set(sgramcounter)

        addtmpscore = 0
        for addgram in addgramcountergood:
            addtmpscore += 1

        addscore_precision = 1
        addscore_recall = 1
        if len(addgramcounter) > 0:
            addscore_precision = addtmpscore / len(addgramcounter)
        if len(addgramcounterall) > 0:
            addscore_recall = addtmpscore / len(addgramcounterall)
        addscore = 0
        if addscore_precision > 0 or addscore_recall > 0:
            addscore = 2 * addscore_precision * addscore_recall / (addscore_precision + addscore_recall)

        return (keepscore, delscore_precision, addscore)

    def normalize(self, sentence, lowercase=True, return_str=True):
        """
        Normalization is requried for the ASSET dataset (one of the primary datasets in sentence simplification) 
        to allow using space to split the sentence.

        Code adapted from the EASSE library written by the authors of the ASSET dataset.
        See: https://github.com/feralvam/easse/blob/580bba7e1378fc8289c663f864e0487188fe8067/easse/utils/preprocessing.py#L7
        """
        if lowercase: sentence = sentence.lower()
        normalized = self.tokenizer(sentence)
        if not return_str: normalized = normalized.split()
        return normalized
    
    def generate_ngrams(self, tokens, n):
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def sari_sent(self, src, tgt, ref):        
        # Generate i-grams for source, target and references
        src_grams, tgt_grams, ref_grams = [src.split(" ")], [tgt.split(" ")], [[rsent.split(" ")] for rsent in ref]
        
        for n in range(2, 5):
            src_grams += [self.generate_ngrams(src_grams[0], n)]
            tgt_grams += [self.generate_ngrams(tgt_grams[0], n)]
            for ref_gram in ref_grams:
                ref_gram += [self.generate_ngrams(ref_gram[0], n)]
        
        # Calculate scores for each gram level and aggregate them
        scores = [self.sari_ngram(src_grams[i], tgt_grams[i], [r[i] for r in ref_grams], len(ref)) for i in range(4)]
        avg_scores = [sum(score) / 4 for score in zip(*scores)] # Average keep, delete, add scores over i-grams
        
        # Calculate the final score as average of averages
        final_score = sum(avg_scores) / 3
        return final_score
    