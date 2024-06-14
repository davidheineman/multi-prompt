from src.metrics import MetricEndpoint

def get_metric(task, ranking_method):
    """
    Depending on the ranking method, get the set of metrics to evaluate
    the candidate set.
    """
    match task:
        case 'simplification': metrics = get_simplification_metric(ranking_method)
        case 'translation': metrics = get_translation_metric(ranking_method)
        case 'code': metrics = get_code_metric(ranking_method)
    return metrics


def get_simplification_metric(ranking_method):
    match ranking_method:
        case 'mbr': metrics = MetricEndpoint('lens', ref=True)
        case 'reranker': metrics = MetricEndpoint('lens_salsa')
        case 'reranker-mbr': metrics = [MetricEndpoint('lens_salsa'), MetricEndpoint('lens', ref=True)]
        case 'multi-turn-mbr': metrics = [MetricEndpoint('lens', ref=True), MetricEndpoint('lens', ref=True)]
        case _: raise ValueError(f'Ranking method {ranking_method} not supported!')
    return metrics


def get_translation_metric(ranking_method):
    match ranking_method:
        case 'mbr': metrics = MetricEndpoint('comet', ref=True)
        case 'reranker': metrics = MetricEndpoint('comet_kiwi')
        case 'reranker-mbr': metrics = [MetricEndpoint('comet_kiwi'), MetricEndpoint('comet', ref=True)]
        case 'multi-turn-mbr': metrics = [MetricEndpoint('comet', ref=True), MetricEndpoint('comet', ref=True)]
        case _: raise ValueError(f'Ranking method {ranking_method} not supported!')
    return metrics


def get_code_metric(ranking_method):
    match ranking_method:
        case 'mbr': metrics = MetricEndpoint('mbr_exec')
        case 'reranker': metrics = MetricEndpoint('code_reranker')
        case 'reranker-mbr': metrics = [MetricEndpoint('code_reranker'), MetricEndpoint('mbr_exec')]
        case 'multi-turn-mbr': metrics = [MetricEndpoint('mbr_exec'), MetricEndpoint('mbr_exec')]
        case _: raise ValueError(f'Ranking method {ranking_method} not supported!')
    return metrics


METRIC_SCORING_FUNCTIONS = {
    "sari": MetricEndpoint("sari", ref=True),
    "lens": MetricEndpoint("lens", ref=True),
    "lens_salsa": MetricEndpoint("lens_salsa"),
    "sle": MetricEndpoint("sle"),
    "bleu": MetricEndpoint("bleu", ref=True),
    "bert_score": MetricEndpoint("bert_score", ref=True),
    "code_bert_score": MetricEndpoint("code_bert_score", ref=True),
    "comet": MetricEndpoint("comet", ref=True),
    "comet_kiwi_xxl": MetricEndpoint("comet_kiwi_xxl"),
    "xcomet": MetricEndpoint("xcomet", ref=True),
    "metricx": MetricEndpoint("metricx", ref=True),
    "metricx_qe": MetricEndpoint("metricx_qe")
}
